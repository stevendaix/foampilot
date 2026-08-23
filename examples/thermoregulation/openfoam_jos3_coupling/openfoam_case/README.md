# Cas OpenFOAM minimal pour le couplage JOS-3

Ce dossier présente la structure minimale d’un cas OpenFOAM destiné à être piloté par FoamPilot. Il s’agit d’un cas pédagogique de configuration : le maillage réel et le solveur thermique doivent être adaptés à votre cas CFD.

## Pourquoi `.out` et `.in` plutôt qu’un CSV ?

Le nom `.out` ne signifie pas « fichier de sortie générique ». C’est le protocole imposé par le function object OpenFOAM `externalCoupled` :

| Fichier | Producteur | Rôle |
|---|---|---|
| `h.out` | OpenFOAM | Coefficient d’échange sortant |
| `air_temperature.out` | OpenFOAM | Température d’air sortante |
| `T.out` | OpenFOAM, si configuré | Température de surface sortante |
| `qJOS3.in` | FoamPilot | Flux nodal renvoyé à OpenFOAM |
| `OpenFOAM.lock` | OpenFOAM/FoamPilot | Synchronisation du pas |

OpenFOAM ne propose pas nativement un échange CSV avec en-tête dans `externalCoupled`. Il attend un format de lignes numériques alignées avec les faces du patch, ainsi qu’un mécanisme de verrouillage. Un CSV peut être ajouté comme format de journalisation ou comme protocole personnalisé, mais il faudrait alors développer un `functionObject` OpenFOAM C++ spécifique qui sache écrire et relire ce CSV. Ce serait moins compatible que le protocole natif.

Les unités ne doivent pas être déduites du nom du fichier. Elles sont définies dans le dictionnaire OpenFOAM par `dimensions`, et dans le contrat FoamPilot. Dans cet exemple, `air_temperature` et `T` sont en kelvins, `h` est en W/m²/K et `qJOS3` est en W/m². Le provider FoamPilot convertit les températures K en °C avant de les transmettre à JOS-3.

Si vous souhaitez un fichier de description lisible avec les unités, le fichier `zone_mapping.csv` joue ce rôle pour le mapping, mais il ne remplace pas les fichiers `.out/.in` synchronisés pendant le calcul.

## OpenFOAM connaît-il automatiquement la zone JOS-3 ?

Non. JOS-3 ne peut pas deviner qu’une face CFD représente la tête, le cou ou le pied. Il possède une liste fixe de 17 segments dans cet ordre :

```text
0 Head          1 Neck          2 Chest         3 Back
4 Pelvis        5 LShoulder     6 LArm          7 LHand
8 RShoulder     9 RArm          10 RHand        11 LThigh
12 LLeg         13 LFoot        14 RThigh       15 RLeg
16 RFoot
```

Le modèle CFD doit donc fournir une correspondance explicite :

```text
face CFD -> zone_id JOS-3 -> aire de face
```

Cette correspondance est stockée dans `zone_mapping.csv`. Le mapping peut être construit de trois manières :

1. le maillage humain est découpé en 17 patches OpenFOAM nommés `Head`, `Neck`, `Chest`, etc. ;
2. le maillage possède un seul patch `humanPatch`, mais chaque face est classée par un champ ou un fichier de mapping ;
3. un maillage de surface est associé à des coordonnées et FoamPilot classe les faces à partir d’une table géométrique préparée par l’utilisateur.

La solution recommandée est un **seul patch humain** pour la condition limite, avec un mapping stable face-vers-zone séparé. Il n’est donc pas nécessaire d’avoir 17 parois OpenFOAM. Il faut en revanche que chaque face du patch soit attribuée à une zone JOS-3, et que l’ordre des faces utilisé par OpenFOAM soit conservé.

## Boucle transitoire

À chaque pas :

```text
OpenFOAM résout le champ fluide et thermique
        ↓
externalCoupled écrit h.out et air_temperature.out
        ↓
FoamPilot lit les faces dans l’ordre OpenFOAM
        ↓
FoamPilot applique zone_id et aire de chaque face
        ↓
DistributedSurfaceNetwork calcule Tsurface[i] et qJOS3[i]
        ↓
FoamPilot écrit qJOS3.in
        ↓
FoamPilot recrée OpenFOAM.lock
        ↓
OpenFOAM reprend le calcul
```

La température de surface `Tsurface[i]` appartient au réseau distribué FoamPilot, et non à JOS-3 original. JOS-3 fournit la température physiologique de référence de la zone ; le réseau distribué possède une température dynamique indépendante par face CFD.

## Structure

```text
openfoam_case/
├── 0/
│   ├── T
│   ├── air_temperature
│   └── h
├── comms/
├── constant/
│   └── polyMesh/          # à remplir avec le vrai maillage
├── system/
│   └── controlDict
├── zone_mapping.csv
└── README.md
```

Le dossier `constant/polyMesh` est volontairement vide dans cet exemple, car il doit être remplacé par le maillage humain et le maillage fluide du cas réel. La configuration `controlDict` montre le point d’accroche `externalCoupled` ; elle doit être ajustée au solveur et aux patches du cas réel.

## Références

[1]: https://doc.openfoam.com/2306/tools/post-processing/function-objects/field/externalCoupled/ "OpenFOAM externalCoupled"

[2]: https://doc.openfoam.com/2306/tools/processing/boundary-conditions/rtm/derived/thermal/externalWallHeatFluxTemperature/ "OpenFOAM externalWallHeatFluxTemperature"
