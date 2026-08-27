from __future__ import annotations

import json
from pathlib import Path

source = Path('/home/ubuntu/foampilot-audit/openfoam13/DTCMoving_Overset_Foundation13/marineInterMeshStencils.json')
output = source.parent / 'background' / 'constant' / 'marineInterMeshStencils'
data = json.loads(source.read_text(encoding='utf-8'))
rows = data['stencils']
lines = [
    'FoamFile',
    '{',
    '    version 2.0;',
    '    format ascii;',
    '    class dictionary;',
    '    object marineInterMeshStencils;',
    '}',
    f"donorRegion {data['donorMesh']};",
    f"acceptorRegion {data['acceptorMesh']};",
    'acceptors',
    '(',
]
for row in rows:
    donors = ' '.join(str(value) for value in row['donorIndices'])
    weights = ' '.join(f"{value:.16g}" for value in row['weights'])
    lines.extend([
        '    {',
        f"        index {row['acceptor']};",
        f'        donorIndices ({donors});',
        f'        weights ({weights});',
        '    }',
    ])
lines.extend([');', ''])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('\n'.join(lines), encoding='utf-8')
print(f'wrote {output} with {len(rows)} stencils')
