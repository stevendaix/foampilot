# Support de cours JOS-3–OpenFOAM

Ce dossier contient le support de cours détaillé en français sur JOS-3 et son couplage thermo-aéraulique avec OpenFOAM dans FoamPilot. Le document couvre les 83 nœuds thermiques, les 17 zones anatomiques, les bilans de chaleur, la conduction, la perfusion sanguine, la convection, le rayonnement, l’évaporation, la respiration, les mécanismes de régulation et le protocole `externalCoupledTemperature`.

Le PDF livré est `main.pdf`. La source éditable est `main.typ` et le thème Typst associé est `report-theme.typ`.

Pour reconstruire le PDF :

```bash
cd course_jos3_openfoam
python3 /home/ubuntu/skills/typst-pdf-maker/scripts/generate_pdf.py main.typ --strict
python3 /home/ubuntu/skills/typst-pdf-maker/scripts/verify_pdf.py main.pdf --profile text-document
```

La compilation validée produit un document de 17 pages. La vérification déterministe donne `PASS` avec six contrôles réussis, zéro avertissement et zéro échec. Les références, l’extrait fourni par l’utilisateur et les URLs consultées sont conservés dans `external_sources.md`.
