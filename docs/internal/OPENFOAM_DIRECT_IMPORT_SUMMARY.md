# OpenFOAM Direct Import Summary

## Overview

Lecture directe d'un cas OpenFOAM avec résultats dans PyVista, sans
passer par l'étape intermédiaire `foamToVTK`.  L'implémentation repose
sur le module existant `foampilot.postprocess.openfoam_direct` et a
été corrigée/testée pour fonctionner avec des cas mono-région et
multi-régions (CHT) réels.

## Methodology

- **Module** : `foampilot/postprocess/openfoam_direct.py`
- **Classes principales** :
  - `OpenFOAMDirectReader` — lecture mono-région vers `pv.UnstructuredGrid`
  - `CHTDirectReader` — lecture CHT vers `pv.MultiBlock`
  - Fonctions de convenance : `read_openfoam()`, `read_cht_openfoam()`
- **Parsing** : lecture ASCII native d'OpenFOAM (`points`, `faces`,
  `owner`, `neighbour`, `boundary`, champs `volScalarField`,
  `volVectorField`, `pointScalarField`, `pointVectorField`).

## Corrections apportées

1. **`_detect_regions`** : détection des régions dans `constant/<region>/polyMesh`
   en plus des répertoires de temps.
2. **`_read_field`** : parsing corrigé de l'en-tête `FoamFile` pour
   extraire la classe de champ (`volVectorField`, `pointScalarField`,
   etc.) et retourner un tuple `(valeurs, est_point)`.
3. **`attach_field`** : attachement automatique en `point_data` ou
   `cell_data` selon la classe du champ, sans devoir spécifier
   `as_point_data` manuellement.
4. **`CHTDirectReader._detect_regions`** : classification heuristique
   fluide/solide par présence du champ `U` dans les répertoires de
   temps.

## Cas testés

| Cas | Type | Régions | Champs validés |
|-----|------|---------|----------------|
| `planarPoiseuille` | mono-région | main | `U` (volVectorField, shape 40x3) |
| `simple_heated_duct` | CHT | fluid, solid | `T` (volScalarField), `U` (volVectorField) |

## Résultats

- Tous les tests du nouveau module passent (`test_direct_openfoam_reader.py` : 13/13).
- Les exports directs existants restent fonctionnels (`test_direct_openfoam_export.py` : 3/3).
- Un script de démonstration est disponible :
  `examples/cht/simple_heated_duct/run_post_direct.py`

## Fichiers modifiés / créés

- `foampilot/src/foampilot/postprocess/openfoam_direct.py` — corrections
  du reader direct
- `foampilot/test/test_direct_openfoam_reader.py` — 13 tests unitaires
- `examples/cht/simple_heated_duct/run_post_direct.py` — démo
  lecture directe + PyVista

## Notes

- Les cellules sont créées avec le type `POLYGON` ; certaines
  opérations PyVista (ex. `slice`) peuvent renvoyer un maillage vide
  selon la géométrie.  Préférer le rendu direct ou le `contour`.
- Les champs sont automatiquement attachés au bon emplacement
  (point/cell) selon leur classe FoamFile.
