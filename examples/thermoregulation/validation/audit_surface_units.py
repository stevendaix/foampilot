from pathlib import Path
import csv
import sys
import numpy as np

ROOT = Path('/home/ubuntu/foampilot')
import importlib.util
import types
JOS_DIR = ROOT / 'foampilot' / 'src' / 'foampilot' / 'physiology' / 'jos3'
pkg = types.ModuleType('jos3')
pkg.__path__ = [str(JOS_DIR)]
sys.modules['jos3'] = pkg
def load(name):
    spec = importlib.util.spec_from_file_location(name, JOS_DIR / (name.split('.')[-1] + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
matrix = load('jos3.matrix')
construction = load('jos3.construction')
localbsa = construction.localbsa
BODY_NAMES = matrix.BODY_NAMES

mapping = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'examples/thermoregulation/makehuman/openfoam_cube_case/zone_mapping_openfoam.csv'
rows = list(csv.DictReader(mapping.open(newline='', encoding='utf-8')))
zone_ids = np.array([int(r['zone_id']) for r in rows])
areas = np.array([float(r['area_m2']) for r in rows])
zone_cfd = np.bincount(zone_ids, weights=areas, minlength=17)

for height, weight, label in [(1.70, 60.0, 'JOS-3 driver supposé 1.70m/60kg'), (1.70, 74.43, '1.70m/poids standard'), (1.72, 74.43, 'JOS-3 défaut')]:
    bsa = localbsa(height, weight, 'dubois')
    print(f'{label}: BSA totale={bsa.sum():.8f} m2')
    print('zone;nom;CFD_m2;JOS3_m2;ratio_CFD_sur_JOS3')
    for i, name in enumerate(BODY_NAMES):
        print(f'{i};{name};{zone_cfd[i]:.8f};{bsa[i]:.8f};{zone_cfd[i]/bsa[i] if bsa[i] else float("nan"):.8f}')
    print(f'ratio total={zone_cfd.sum()/bsa.sum():.8f}\n')
print(f'CSV={mapping}')
print(f'CFD total={areas.sum():.8f} m2; faces={areas.size}; min={areas.min():.8g}; max={areas.max():.8g}')
print('Protocol dimensions: data.out = area[m2], T[K], qDot[W/m2], htc[W/m2/K]; data.in = T[K], snGrad[K/m], valueFraction[-].')
