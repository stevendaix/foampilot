# VOF vers DPM avec OpenFOAM 13

Cette page décrit l’ensemble du projet VOF–DPM intégré à foampilot : convertisseur Python, sources C/C++ natifs OpenFOAM 13, modèles `fvModel` incompressible et compressible, cas de validation et génération de la note technique PDF.

> L’implémentation actuelle distingue l’extraction offline et le couplage solver–cloud. Elle ne doit pas être présentée comme une transition temps réel conservative tant que le volume VOF converti n’est pas retiré et que le parcel n’est pas inséré de manière transactionnelle.

## 1. Prérequis et installation

Les commandes suivantes supposent une installation OpenFOAM 13 sous Ubuntu dans `/opt/openfoam13`.

```bash
sudo apt update
sudo apt install -y git build-essential python3 python3-pip
. /opt/openfoam13/etc/bashrc
foamVersion
```

Depuis la racine du dépôt foampilot, installer les dépendances Python :

```bash
cd foampilot
sudo pip3 install -r requirements.txt
sudo pip3 install pytest
```

Pour utiliser uniquement le convertisseur, NumPy et pytest suffisent pour les tests ciblés. Le chargement complet du package foampilot peut demander les dépendances optionnelles de géométrie et de post-traitement.

## 2. Où se trouvent les fichiers

Le projet complet VOF–DPM OpenFOAM 13 se trouve dans `foampilot/examples/openfoam13/vof_to_dpm/`.

| Chemin | Contenu |
|---|---|
| `src/foampilot/utilities/vof_to_dpm.py` | Lecteur OpenFOAM ASCII, extraction des composantes et écriture des sorties |
| `test/test_vof_to_dpm.py` | Tests unitaires Python |
| `examples/course_vof_to_dpm.py` | Exercice Python pédagogique |
| `examples/generate_vof_to_dpm_technical_note.py` | Générateur de la note PDF |
| `src/foampilot/report/typst_pdf.py` | Moteur de génération Typst utilisé par la note |
| `examples/openfoam13/vof_to_dpm/applications/vofToDpm` | Extracteur C++ offline |
| `examples/openfoam13/vof_to_dpm/applications/incompressibleVoFClouds` | Pont `fvModel` incompressible |
| `examples/openfoam13/vof_to_dpm/applications/compressibleVoFClouds` | Pont `fvModel` compressible |
| `examples/openfoam13/vof_to_dpm/test/openfoam13` | Cas OpenFOAM 13 et scripts `Allrun` |
| `docs/fr/vof_to_dpm_technical_note.pdf` | Note technique générée |

Les sources historiques complètes de `statisticalDPMFoam`, notamment les fichiers `.C`, `.H`, `Make/files` et `Make/options`, sont sous `examples/openfoam13/vof_to_dpm/statisticalDPMFoam/`.

## 3. Exécuter les tests Python

Depuis le répertoire `foampilot` :

```bash
PYTHONPATH=src/foampilot/utilities python -m pytest -q test/test_vof_to_dpm.py
```

Les tests vérifient les fragments séparés et connectés, la pondération `alpha × V`, les indices invalides, les filtres, la lecture de champs ASCII OpenFOAM et l’écriture des sorties.

Lancer l’exercice synthétique du cours :

```bash
PYTHONPATH=src python examples/course_vof_to_dpm.py
```

Le programme affiche le nombre de fragments, les volumes initial et converti, le résidu de volume et la quantité de mouvement pondérée avant et après extraction.

## 4. Compiler les composants C/C++ OpenFOAM

Dans chaque nouveau terminal :

```bash
. /opt/openfoam13/etc/bashrc
cd foampilot/examples/openfoam13/vof_to_dpm
```

Compiler les trois composants du couplage :

```bash
wmake applications/vofToDpm
wmake applications/incompressibleVoFClouds
wmake applications/compressibleVoFClouds
```

Les fichiers `Make/files` et `Make/options` sont conservés avec chaque composant. Les objets produits dans `Make/linux64*` sont générés localement et ne sont pas versionnés.

Pour compiler la famille de solveurs `statisticalDPMFoam` regroupée dans foampilot :

```bash
cd examples/openfoam13/vof_to_dpm/statisticalDPMFoam
./Allwmake
```

## 5. Lancer l’extracteur C++ offline

L’utilitaire natif lit un cas sériel, un champ `alpha`, éventuellement `U`, et la connectivité du maillage. Exemple :

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/vofToDpmSingleCell
. /opt/openfoam13/etc/bashrc
../../../../applications/vofToDpm/Make/linux64GccDPInt32Opt/vofToDpm \
    -alpha alpha.liquid -U U -threshold 0.5 -rhoLiquid 1000
```

Le chemin exact de l’exécutable peut varier selon le compilateur, la précision et les options OpenFOAM. Les fichiers produits contiennent les positions, les propriétés des fragments et le rapport des volumes sélectionné, converti et rejeté. Le volume liquide est calculé par `sum(alpha_i × V_i)` sans renormalisation.

## 6. Lancer le cas incompressible

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/incompressibleVoFCloudsDamBreak
./Allrun
```

Le script prépare le cas, active `fvModels` et le prédicteur de quantité de mouvement, charge `incompressibleVoFClouds` et vérifie l’activité du cloud. L’injection actuelle est manuelle : elle valide le chemin solver–cloud, mais pas encore la conversion automatique d’un fragment VOF.

## 7. Lancer le cas compressible

```bash
cd foampilot/examples/openfoam13/vof_to_dpm/test/openfoam13/compressibleVoFCloudsDamBreak
./Allrun
```

Ce smoke test valide la sélection runtime de `compressibleVoFClouds`, l’évolution mécanique du cloud et le retour de quantité de mouvement. Il ne constitue pas encore une transition thermodynamiquement conservative : la masse, l’enthalpie et la pression nécessitent un transfert dédié.

## 8. Générer la note technique PDF

Le générateur utilise les classes `ScientificDocument` et `TypstRenderer` de foampilot :

```bash
cd foampilot
python examples/generate_vof_to_dpm_technical_note.py
```

Les fichiers générés sont placés dans `report/` lorsque la commande est lancée depuis la racine du dépôt :

```text
report/vof_to_dpm_technical_note.pdf
report/vof_to_dpm_technical_note.typ
report/vof_to_dpm.bib
```

La note détaille les critères de transition, les équations de conservation, l’audit des simplifications et l’architecture recommandée pour une version temps réel.

## 9. Portée scientifique actuelle

Le convertisseur Python et l’utilitaire C++ calculent correctement le volume d’une composante, son centroïde, sa vitesse moyenne pondérée et son diamètre sphérique équivalent. Les deux `fvModel` font évoluer un `parcelCloudList` et renvoient son terme mécanique dans l’équation de quantité de mouvement.

Une transition automatique de production nécessite encore une consommation bornée de `alpha`, une insertion dynamique des parcels, des identifiants stables de fragments, une réconciliation MPI, une prévention des doubles conversions et, en compressible, un transfert cohérent de masse et d’énergie.

## 10. Documentation multilingue

| Langue | Guide installation/exécution | Supports détaillés |
|---|---|---|
| English | `docs/en/vof_to_dpm_openfoam13.md` | `docs/en/vof_to_dpm.md` |
| Français | `docs/fr/vof_to_dpm_openfoam13.md` | `docs/fr/cours_vof_to_dpm.md`, `docs/fr/audit_implementation_vof_to_dpm.md` |
| 中文 | `docs/zh/vof_to_dpm_openfoam13.md` | `docs/zh/vof_to_dpm.md` |
