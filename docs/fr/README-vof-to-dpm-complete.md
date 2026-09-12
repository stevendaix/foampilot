# Projet complet VOF-to-DPM dans foampilot

Ce répertoire regroupe l'ensemble des livrables du projet VOF-to-DPM OpenFOAM 13.

- `../../src/foampilot/utilities/vof_to_dpm.py` : convertisseur Python et lecteur ASCII OpenFOAM.
- `../../test/test_vof_to_dpm.py` : tests unitaires du convertisseur.
- `../../examples/course_vof_to_dpm.py` : exercice Python pédagogique.
- `../../examples/generate_vof_to_dpm_technical_note.py` : génération de la note PDF.
- `../../src/foampilot/report/typst_pdf.py` : moteur de rapport Typst corrigé.
- `../../examples/openfoam13/vof_to_dpm/applications` : utilitaire C++ `vofToDpm` et `fvModel` incompressible/compressible.
- `../../examples/openfoam13/vof_to_dpm/test` : cas OpenFOAM 13 et scripts `Allrun`.
- `vof_to_dpm_technical_note.pdf` : note technique générée avec foampilot.
- `cours_vof_to_dpm.md` et `audit_implementation_vof_to_dpm.md` : supports pédagogiques et audit.

Les objets de compilation OpenFOAM et les répertoires `processor*` ne sont volontairement pas importés ; ils sont reproductibles avec les fichiers `Make` et les scripts de validation.
