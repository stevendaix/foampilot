from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def main() -> None:
    path = Path('/tmp/medical_build_complex_prebuild/analysis_contract.json')
    data = json.loads(path.read_text())
    branch = next(b for b in data['branches'] if b['branch_id'] == 2)
    rows = []
    for i, section in enumerate(branch['sections']):
        p = np.asarray(section['points'], dtype=float)
        phase = np.asarray(section.get('phase_locked_points', p), dtype=float)
        c = np.asarray(section['center'], dtype=float)
        t = np.asarray(section['tangent'], dtype=float)
        radial = np.linalg.norm(p - c, axis=1)
        phase_radial = np.linalg.norm(phase - c, axis=1)
        rows.append({
            'index': i,
            'station_id': section.get('station_id', section.get('point_id')),
            'abscissa': section.get('abscissa'),
            'center': c.tolist(),
            'point_count': len(p),
            'radius_min': float(radial.min()),
            'radius_median': float(np.median(radial)),
            'radius_max': float(radial.max()),
            'radius_std': float(radial.std()),
            'phase_radius_median': float(np.median(phase_radial)),
            'phase_radius_max': float(phase_radial.max()),
            'phase_radius_std': float(phase_radial.std()),
            'center_jump': None if i == 0 else float(np.linalg.norm(c - np.asarray(branch['sections'][i-1]['center'], dtype=float))),
            'tangent': t.tolist(),
        })
    out = Path('examples/medical_build/outputs/reconstructed_local_deformation/diagnostics/central_cylinder_sections.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'branch_id': 2, 'sections': rows}, indent=2))
    suspicious = sorted(rows, key=lambda r: r['radius_median'], reverse=True)[:10]
    print(json.dumps({'branch_id': 2, 'section_count': len(rows), 'largest_sections': suspicious}, indent=2))


if __name__ == '__main__': main()
