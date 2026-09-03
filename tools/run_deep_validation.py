from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import shutil

root = Path(__file__).resolve().parents[1]
foampilot = root / 'foampilot'
report = root / 'deep-validation-report.md'
foam_env = r'''if [ -n "${FOAM_BASHRC:-}" ] && [ -f "$FOAM_BASHRC" ]; then source "$FOAM_BASHRC"; elif [ -n "${WM_PROJECT_DIR:-}" ] && [ -f "$WM_PROJECT_DIR/etc/bashrc" ]; then source "$WM_PROJECT_DIR/etc/bashrc"; else echo "FOAM_BASHRC or WM_PROJECT_DIR is required" >&2; exit 2; fi'''
lines = ['# Validation approfondie Foampilot / OpenFOAM 13', '', '| Test | Résultat | Détail |', '|---|---:|---|']

def add(name, ok, detail):
    lines.append(f'| {name} | {"PASS" if ok else "FAIL"} | {detail.replace(chr(10), " ")} |')

r = subprocess.run(['pytest', '-q', 'test/test_multiphysics_integration.py'], cwd=root, text=True, capture_output=True)
add('Tests unitaires multiphysiques', r.returncode == 0, (r.stdout + r.stderr).strip()[-500:])

module_path = foampilot / 'src/foampilot/multiphysics/integration.py'
spec = importlib.util.spec_from_file_location('fpi', module_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
config = mod.MultiphysicsConfiguration(('openhfdib_dem', 'libacoustics'))
assets = root / 'validation' / 'generated-sedifoam-acoustics'
assets.mkdir(parents=True, exist_ok=True)
manifest, dictionary = config.write_case_assets(assets)
add('Génération manifeste Foampilot', manifest.exists() and dictionary.exists(), str(manifest))
add('Profil openHFDIB-DEM/libAcoustics', config.modules == ('openhfdib_dem', 'libacoustics'), 'combinaison autorisée avec openHFDIB-DEM + acoustique')
config_sedi = mod.MultiphysicsConfiguration(('sedifoam', 'libacoustics'))
plan = config_sedi.build_plan if hasattr(config_sedi, 'build_plan') else mod.build_plan(config_sedi, root)
add('Profil sediFoam/libAcoustics', config_sedi.modules == ('sedifoam', 'libacoustics'), 'combinaison autorisée avec sediFoam + acoustique')
add('Chemins vendorisés', all(Path(item['source']).exists() for item in plan), json.dumps(plan))

manifest_data = json.loads(manifest.read_text(encoding='utf-8'))
add('Validation JSON manifeste', manifest_data['openfoam'] == {'distribution': 'Foundation', 'version': '13'}, 'JSON valide et version Foundation 13')
rr = subprocess.run(['bash', '-lc', f'{foam_env} && foamDictionary -entry openfoamVersion {dictionary}'], text=True, capture_output=True)
add('Parsing dictionnaire OpenFOAM', rr.returncode == 0 and '13' in (rr.stdout + rr.stderr), (rr.stdout + rr.stderr).strip()[-300:])

rr = subprocess.run(['bash', '-lc', f'{foam_env} && foamVersion'], text=True, capture_output=True)
foam_version_output = (rr.stdout + rr.stderr).strip()
add('Version OpenFOAM', rr.returncode == 0 and foam_version_output == 'OpenFOAM-13', foam_version_output)

foam_user_libbin = subprocess.check_output(['bash', '-lc', f'{foam_env} && printf "%s" "$FOAM_USER_LIBBIN"'], text=True).strip()
lib = Path(foam_user_libbin) / 'libHFDIBDEM.so'
for exe in ['HFDIBDEMFoam', 'pimpleHFDIBFoam']:
    rr = subprocess.run(['bash', '-lc', f'{foam_env} && {exe} -help'], text=True, capture_output=True)
    add(f'{exe} -help', rr.returncode == 0, (rr.stdout + rr.stderr).strip()[-200:])
add('Bibliothèque HFDIBDEM', lib.exists(), str(lib))
acoustic_lib = Path(foam_user_libbin) / 'libAcoustics.so'
sediment_lib = Path(foam_user_libbin) / 'libLagrangianInterfacialModels.so'
add('Bibliothèque libAcoustics OF13', acoustic_lib.exists(), str(acoustic_lib))
add('Bibliothèque dragModels sediFoam OF13', sediment_lib.exists(), str(sediment_lib))
for label, library, symbols in [
    ('Symboles Curle/FW-H', acoustic_lib, ['CurleAnalogy', 'FfowcsWilliamsHawkings']),
    ('Symboles dragModels', sediment_lib, ['ErgunWenYu', 'SyamlalOBrien']),
]:
    if library.exists():
        symbol_text = subprocess.run(['nm', '-D', '--defined-only', str(library)], text=True, capture_output=True).stdout
        add(label, all(symbol in symbol_text for symbol in symbols), ', '.join(symbols))
    else:
        add(label, False, f'artefact absent: {library}')

case_src = Path(os.environ.get('MULTIPHYSICS_VALIDATION_CASE', str(root / 'validation' / 'normalForce_OF13')))
case = root / 'validation' / 'normalForce_OF13_deep'
if not case_src.exists():
    add('Cas DEM HFDIB', False, f'cas de validation absent: {case_src}; fournir MULTIPHYSICS_VALIDATION_CASE')
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    raise SystemExit(1)
if case.exists():
    shutil.rmtree(case)
shutil.copytree(case_src, case)
control = case / 'system/controlDict'
text = control.read_text()
import re
text = re.sub(r'^startFrom\s+\S+;', 'startFrom       startTime;', text, flags=re.MULTILINE)
text = re.sub(r'^endTime\s+\S+;', 'endTime         0.005;', text, flags=re.MULTILINE)
text = re.sub(r'^deltaT\s+\S+;', 'deltaT          1e-4;', text, flags=re.MULTILINE)
control.write_text(text)
rr = subprocess.run(['bash', '-lc', f'{foam_env} && checkMesh -case {case} -latestTime'], text=True, capture_output=True)
add('checkMesh cas DEM', rr.returncode == 0, (rr.stdout + rr.stderr).strip()[-500:])
rr = subprocess.run(['bash', '-lc', f'{foam_env} && foamDictionary -entry dynamicFvMesh {case}/constant/dynamicMeshDict'], text=True, capture_output=True)
add('dynamicMeshDict statique', rr.returncode == 0 and 'staticFvMesh' in (rr.stdout + rr.stderr), (rr.stdout + rr.stderr).strip())
log = root / 'validation-normalForce-of13-deep.log'
with log.open('w') as fh:
    rr = subprocess.run(['bash', '-lc', f'{foam_env} && HFDIBDEMFoam -case {case} -noFunctionObjects'], stdout=fh, stderr=subprocess.STDOUT, text=True)
log_text = log.read_text(errors='replace')
steps = log_text.count('Time = ')
updates = log_text.count('updated HFDIBDEM')
add('Exécution DEM prolongée', rr.returncode == 0 and steps >= 45 and updates >= 45, f'code={rr.returncode}, pas={steps}, mises à jour={updates}')
for marker in ['Creating immersed body based on: sphere_Top1', 'Creating immersed body based on: sphere_Bot1', 'ExecutionTime =']:
    add(f'Sortie DEM: {marker}', marker in log_text, 'présent' if marker in log_text else 'absent')

report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(report)
