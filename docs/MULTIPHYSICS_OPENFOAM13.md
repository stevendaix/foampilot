# Intégration multiphysique OpenFOAM Foundation 13

Cette extension ajoute à Foampilot un **contrat d’intégration explicite** pour `sediFoam`, `openHFDIB-DEM` et `libAcoustics`. Elle ne vend pas un faux « copier-coller » : les trois projets ciblent des générations et parfois des distributions OpenFOAM différentes. Foampilot conserve donc les sources externes hors du paquet Python, génère un manifeste auditable et refuse les combinaisons physiques ambiguës.

## Architecture retenue

| Module | Physique | Référence amont | Statut OF13 dans cette PR |
| --- | --- | --- | --- |
| `sediFoam` | CFD–DEM avec LAMMPS, transport sédimentaire | `master` | Profil Foampilot et contrat de champs ; portage C++/LAMMPS à compiler séparément |
| `openHFDIB-DEM` | CFD–DEM immersed-boundary, particules de forme arbitraire | `master`, développé pour OpenFOAM v8 | Profil Foampilot ; l’ancienne API doit être portée vers OF13 |
| `libAcoustics` | sources acoustiques et FW-H | `v2512`, OpenFOAM+ | Profil Foampilot ; `v2512` est une référence ESI et non une compatibilité Foundation garantie |

Les deux backends DEM sont **mutuellement exclusifs dans un cas**. L’acoustique peut être ajoutée à l’un ou l’autre backend, mais elle doit être traitée comme une physique de mesure/post-traitement et non comme un solveur DEM concurrent.

## Utilisation

```python
from foampilot import MultiphysicsConfiguration

config = MultiphysicsConfiguration(("openhfdib_dem", "libacoustics"))
config.write_case_assets("./case")
```

La commande produit `system/foampilotMultiphysics.json`, destiné à l’audit et à l’orchestration, et `system/foampilotMultiphysics`, dictionnaire OpenFOAM lisible. Les champs requis sont calculés automatiquement ; notamment `nu` doit continuer à être écrit explicitement dans `constant/transportProperties` par le générateur de propriétés de Foampilot.

## Vérifications exécutées

L’environnement a été installé depuis le dépôt officiel OpenFOAM Foundation pour Ubuntu 24.04. La commande `foamVersion` retourne `OpenFOAM-13`. Les tests ciblés de cette extension passent avec **4 tests réussis**.

Un premier build réel de `openHFDIB-DEM` a également été lancé avec OF13. Il s’arrête avant la compilation métier car le code inclut `fvCFD.H`, en-tête de commodité absent de l’arborescence OF13 installée. Ce résultat est volontairement conservé comme signal de portage : remplacer l’en-tête par une simple copie ne constitue pas un portage complet, car les classes de maillage, d’interpolation, de champs et de solveur doivent ensuite être vérifiées. `sediFoam` ajoute en plus une dépendance forte à LAMMPS et à son interface C++. `libAcoustics` possède sa propre bibliothèque et des tests dans `v2512`, mais ses dictionnaires et conventions sont ceux d’OpenFOAM+.

## Étapes de portage C++ restantes

Le portage complet côté solveurs doit être mené dans l’ordre suivant : créer une couche d’includes OF13, adapter les options `Make/options` aux bibliothèques Foundation 13, compiler chaque bibliothèque sans parallélisme, corriger les changements de types et de signatures, puis exécuter un cas minimal et un cas de validation scientifique. Après chaque étape, le journal de build doit être conservé dans le CI ou dans les artefacts de la PR. L’adaptateur Foampilot fournit déjà le point unique de configuration afin que ces builds puissent être branchés sans dupliquer la logique des cas.

## Références

[1]: https://github.com/xiaoh/sediFoam "Dépôt sediFoam"

[2]: https://github.com/techMathGroup/openHFDIB-DEM "Dépôt openHFDIB-DEM"

[3]: https://github.com/unicfdlab/libAcoustics/tree/v2512 "Branche v2512 de libAcoustics"

[4]: https://openfoam.org/download/13-ubuntu/ "Installation officielle OpenFOAM 13 pour Ubuntu"
