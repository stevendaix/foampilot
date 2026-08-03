# Refroidissement d'Électronique CHT

Exemple foampilot démontrant la simulation de transfert thermique conjugué
(CHT) multi-régions pour un composant électronique (puce sur dissipateur
à ailettes) refroidi par convection forcée.

## Description

Ce cas modélise :

| Région | Matériau | Type | Rôle |
|--------|----------|------|------|
| **chip** | Silicium | Solide | Source de chaleur volumique |
| **heatsink** | Aluminium | Solide | Conduction et dissipation par ailettes |
| **fluid** | Air | Fluide | Convection forcée |

Le maillage est généré paramétriquement avec Gmsh (OpenCASCADE) et
exporté directement vers le format polyMesh OpenFOAM via
`DirectOpenFOAMExporter`, sans passer par `gmshToFoam`.

## Workflow

1. **Définition paramétrique** — dimensions de la puce, du dissipateur
   (base + ailettes), et du domaine fluide
2. **Construction Gmsh** — opérations booléennes (`fuse`, `cut`) pour
   créer les volumes imbriqués fluide/solide
3. **Groupes physiques** — attribution des noms de régions (`air`,
   `chip`, `heatsink`) et des patches frontières (`inlet`, `outlet`,
   `wall`, `top`, `bottom`)
4. **Maillage** — génération tétraédrique avec raffinement local près
   de la puce et des ailettes
5. **Export direct** — `DirectOpenFOAMExporter.export_multi_region()`
   écrit `constant/fluid/polyMesh`, `constant/chip/polyMesh`,
   `constant/heatsink/polyMesh`
6. **Configuration CHT** — `ChtSolver` avec `chtMultiRegionFoam`,
   `FluidRegion`, `SolidRegion`, et `CoupledInterface` pour les
   interfaces fluide-solide

## Comparaison avec des cas similaires

| Cas | Source | Description | Différences |
|-----|--------|-------------|-------------|
| **ChipHX** | KIT / BwUniCluster | Plaque (puce) + cylindres (pins) dans OpenFOAM | Utilise `blockMesh` + `snappyHexMesh` ; géométrie STL ; solver `chtMultiRegionSimpleFoam` |
| **circuitBoardCooling** | OpenFOAM tutorials | Carte PCB avec baffles 3D pour CHT | Utilise des baffles thermiques 3D au lieu de régions solides séparées |
| **Rectangular Fins** | SimScale | Validation CHT dissipateur à ailettes rectangulaires | Outil SimScale (OpenFOAM sous-jacent) ; maillage automatisé ; comparaison avec corrélation analytique R_ja |
| **LED COP** | SimScale | Gestion thermique LED puce sur plaque | Puissance volumique uniforme ; comparaison avec résultats expérimentaux |
| **Gin Tonic** | Holzmann CFD | Cas d'entraînement CHT OpenFOAM | Utilise `snappyHexMesh` + `createPatch` ; cas multi-régions avec `chtMultiRegionFoam` |
| **ConjugateHeatTransfer-OpenFOAM** | tandise/GitHub | Comparaison de 3 algorithmes CHT | Solver `multiChtFoam` (foam-extend) ; approche monolithique et partitionnée |

L'exemple `electronique` se distingue par :
- La **construction paramétrique** de la géométrie (nombre d'ailettes,
  épaisseur, hauteur) via l'API Python Gmsh
- L'**export direct** sans fichier intermédiaire `.msh`
- L'utilisation de **`ChtSolver`** de foampilot pour l'écriture
  automatique des fichiers de région et des interfaces CHT

## Cas de comparaison

Des cas variant sont disponibles pour comparer l'impact de
différents paramètres sur les résultats CHT :

| Cas | Paramètre modifié | Valeur |
|-----|-------------------|--------|
| `electronique_few_fins` | Nombre d'ailettes | 3 (au lieu de 5) |
| `electronique_high_velocity` | Vitesse d'entrée | 2 m/s (au lieu de 1 m/s) |
| `electronique_copper_heatsink` | Matériau du dissipateur | Cuivre (k=401, au lieu d'aluminium k=205) |

Chaque cas possède son propre `run.py` utilisant les mêmes
méthodes partagées de `GmshMesher`.

```bash
cd examples/cht/electronique
python run.py
```

Le script génère tous les fichiers de cas dans le répertoire courant.

Pour lancer la simulation (nécessite OpenFOAM 13 installé) :

```bash
chtMultiRegionFoam
```

## Paramètres modifiables

Les paramètres sont définis en haut du script `run.py` :

- `chip_lx`, `chip_ly`, `chip_lz` — dimensions de la puce
- `hs_base_lx`, `hs_base_ly`, `hs_base_lz` — dimensions de la base du dissipateur
- `fin_height`, `fin_thickness`, `fin_spacing`, `n_fins` — géométrie des ailettes
- `domain_lx`, `domain_ly`, `domain_lz` — taille du domaine fluide
- `inlet_velocity`, `inlet_temperature` — conditions d'entrée
- `chip_heat_flux` — flux thermique de la puce
- `chip_kappa/rho/cp`, `hs_kappa/rho/cp` — propriétés matériaux
- `lc_min`, `lc_max`, `lc_fin`, `lc_chip` — paramètres de maillage