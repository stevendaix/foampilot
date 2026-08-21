from pathlib import Path
import numpy as np

path = Path('/usr/share/makehuman-community/data/3dobjs/base.npz')
data = np.load(path, allow_pickle=True)
print(data.files)
for name in data.files:
    value = data[name]
    print(name, value.shape, value.dtype)
coord = data['coord'].astype(float)
print('bounds=', coord.min(axis=0), coord.max(axis=0))
print('face-index-max=', data['fvert'].max(), 'coord-count=', coord.shape[0])
