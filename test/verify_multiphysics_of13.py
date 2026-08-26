from pathlib import Path
from foampilot import MultiphysicsConfiguration, check_openfoam13

root = Path('/tmp/foampilot-of13-case')
root.mkdir(parents=True, exist_ok=True)
manifest, dictionary = MultiphysicsConfiguration(('openhfdib_dem', 'libacoustics')).write_case_assets(root)
print(check_openfoam13())
print(manifest)
print(dictionary)
