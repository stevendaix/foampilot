# Urban CFD — Outils

## Candidates

| Outil | URL | Licence | Python | Linux | Décision |
|-------|-----|---------|--------|-------|----------|
| `foampilot` | https://github.com/stevendaix/foampilot | - | ✅ | ✅ | **USE** |
| `WindAnalysis` | interne | - | ✅ | ✅ | **USE** |
| `snappyHexMesh` | OpenFOAM natif | GPL | ❌ | ✅ | **WRAP** |

## Recommandation

- **Gmsh** : backend principal pour MVP (via `GmshQuarterBuilder`)
- **snappyHexMesh** : backend futur pour gros quartiers (Phase 7)
- **WindAnalysis** : déjà intégré dans `foampilot`

## Notes

- Pas d'outil Python dédié "urban CFD geometry" retenu.
- La couche `foampilot.urban` est elle-même la réponse.
