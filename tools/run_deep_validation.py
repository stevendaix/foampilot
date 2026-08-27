from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import shutil

root = Path('/home/ubuntu/foam-integration')
foampilot = root / 'foampilot'
report = root / 'deep-validation-report.md'
lines = ['# Validation approfondie Foampilot / OpenFOAM 13', '', '| Test | Résultat | Détail |', '|---|---:|---|']

def add(name, ok, detail):
    lines.append(f'| {name} | {"PASS" if ok else "FAIL"} | {detail.replace(chr(10), " ")} |')

r = subprocess.run(['pytest', '-q', 'test/test_multiphysics_integration.py'], cwd=foampilot, text=True, capture_output=True)
add('Tests unitaires multiphysiques', r.returncode == 0, (r.stdout + r.stderr).strip()[-500:])

module_path = foampilot / 'foampilot/src/foampilot/multiphysics/integration.py'
spec = importlib.util.spec_from_file_location('fpi', module_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
config = mod.MultiphysicsConfiguration(('openhfdib_dem', 'libacoustics'))
assets = root / 'validation' / 'generated-sedifoam-acoustics'
assets.mkdir(parents=True, exist_ok=True)
manifest, dictionary = config.write_case_assets(assets)
add('Génération manifeste Foampilot', manifest.exists() and dictionary.exists(), str(manifest))
add('Profil sediFoam/libAcoustics', config.modules == ('openhfdib_dem', 'libacoustics'), 'combinaison autorisée avec openHFDIB-DEM + acoustique')

manifest_data = json.loads(manifest.read_text(encoding='utf-8'))
add('Validation JSON manifeste', manifest_data['openfoam'] == {'distribution': 'Foundation', 'version': '13'}, 'JSON valide et version Foundation 13')
rr = subprocess.run(['bash', '-lc', f'. /opt/openfoam13/etc/bashrc && foamDictionary -entry openfoamVersion {dictionary}'], text=True, capture_output=True)
add('Parsing dictionnaire OpenFOAM', rr.returncode == 0 and '13' in (rr.stdout + rr.stderr), (rr.stdout + rr.stderr).strip()[-300:])

rr = subprocess.run(['bash', '-lc', '. /opt/openfoam13/etc/bashrc && foamVersion'], text=True, capture_output=True)
foam_version_output = (rr.stdout + rr.stderr).strip()
add('Version OpenFOAM', rr.returncode == 0 and foam_version_output == 'OpenFOAM-13', foam_version_output)

lib = Path('/home/ubuntu/OpenFOAM/root-13/platforms/linux64GccDPInt32Opt/lib/libHFDIBDEM.so')
for exe in ['HFDIBDEMFoam', 'pimpleHFDIBFoam']:
    rr = subprocess.run(['bash', '-lc', f'. /opt/openfoam13/etc/bashrc && {exe} -help'], text=True, capture_output=True)
    add(f'{exe} -help', rr.returncode == 0, (rr.stdout + rr.stderr).strip()[-200:])
add('Bibliothèque HFDIBDEM', lib.exists(), str(lib))

case_src = root / 'validation' / 'normalForce_OF13'
case = root / 'validation' / 'normalForce_OF13_deep'
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
rr = subprocess.run(['bash', '-lc', f'. /opt/openfoam13/etc/bashrc && checkMesh -case {case} -latestTime'], text=True, capture_output=True)
add('checkMesh cas DEM', rr.returncode == 0, (rr.stdout + rr.stderr).strip()[-500:])
rr = subprocess.run(['bash', '-lc', f'. /opt/openfoam13/etc/bashrc && foamDictionary -entry dynamicFvMesh {case}/constant/dynamicMeshDict'], text=True, capture_output=True)
add('dynamicMeshDict statique', rr.returncode == 0 and 'staticFvMesh' in (rr.stdout + rr.stderr), (rr.stdout + rr.stderr).strip())
log = root / 'validation-normalForce-of13-deep.log'
with log.open('w') as fh:
    rr = subprocess.run(['bash', '-lc', f'. /opt/openfoam13/etc/bashrc && HFDIBDEMFoam -case {case} -noFunctionObjects'], stdout=fh, stderr=subprocess.STDOUT, text=True)
log_text = log.read_text(errors='replace')
steps = log_text.count('Time = ')
updates = log_text.count('updated HFDIBDEM')
add('Exécution DEM prolongée', rr.returncode == 0 and steps >= 45 and updates >= 45, f'code={rr.returncode}, pas={steps}, mises à jour={updates}')
for marker in ['Creating immersed body based on: sphere_Top1', 'Creating immersed body based on: sphere_Bot1', 'ExecutionTime =']:
    add(f'Sortie DEM: {marker}', marker in log_text, 'présent' if marker in log_text else 'absent')

report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(report)
