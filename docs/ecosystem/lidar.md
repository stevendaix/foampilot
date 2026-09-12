# LiDAR — Outils

## Candidates

| Outil | URL | Licence | Python | Linux | Décision |
|-------|-----|---------|--------|-------|----------|
| `lazrs` | https://github.com/laz-rs/laz-rs-python | MIT | ✅ | ✅ | **WRAP** |
| `laspy` | https://github.com/laspy/laspy | BSD-2-Clause | ✅ | ✅ | **WRAP** |
| `PDAL` | https://github.com/PDAL/python | NOASSERTION | ✅ | ✅ | **WRAP** |

## Recommandation

- **Lecture LAS/LAZ** : `laspy` + backend `lazrs`
- **Traitements simples** : `laspy` + numpy
- **Pipelines complexes** (classification, DTM, segmentation) : `PDAL` + `filters.python` si nécessaire

## Notes

- `laspy` 2.7.0 (jan 2026) supporte Python 3.14, LAS 1.5, COPC
- `lazrs` 0.8.x requis pour LAZ
- `PDAL` nécessite une installation système + binding Python
