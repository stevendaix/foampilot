# Couplage Cantera–OpenFOAM 13

Cette contribution fournit deux interfaces complémentaires. L’adaptateur Python `foampilot.coupling.cantera_openfoam` réalise un échange de fichiers CSV, utile pour l’orchestration et les tests. Le programme C++ `third_party/cantera-openfoam13/canteraFoam` est un **bridge in-process OpenFOAM/Cantera** : il est compilé avec `wmake`, lit `T`, `p` et `canteraProperties` dans un cas OpenFOAM 13, évalue l’état Cantera cellule par cellule et écrit `canteraThermo.csv`. Il ne modifie pas encore les équations d’espèces ou d’énergie d’un solveur réactif ; ce n’est donc pas un portage de `reactingFoam` ni une injection de termes sources chimiques.

## Versions validées

La combinaison exécutée et validée dans cette branche est :

| Composant | Version ou emplacement |
|---|---|
| Ubuntu | 24.04 |
| OpenFOAM Foundation | 13, environnement `/opt/openfoam13/etc/bashrc` |
| Cantera Python | 3.2.0 dans l’environnement Python utilisé par `python3` |
| Cantera C++ | 4.0.0a2, installé sous `$HOME/.local/cantera` |
| Compilateur C++ | GCC avec standard C++20 pour les headers Cantera 4 |

Le binaire C++ et la validation doivent utiliser une installation C++ de Cantera cohérente avec `CANTERA_ROOT`. Une installation Python seule ne suffit pas pour compiler `canteraFoam`.

## Installation complète depuis un checkout propre

### 1. OpenFOAM Foundation 13

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common wget gnupg
sudo sh -c 'wget -qO- https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc'
sudo add-apt-repository -y 'http://dl.openfoam.org/ubuntu main dev'
sudo apt-get update
sudo apt-get install -y --no-install-recommends openfoam13
```

Dans chaque nouveau shell de travail, définir le chemin de l’environnement OpenFOAM. Le lanceur n’impose pas `/opt/openfoam13` et accepte aussi une autre installation Foundation 13 :

```bash
export FOAM_BASHRC=/opt/openfoam13/etc/bashrc
source "$FOAM_BASHRC"
foamVersion
```

La commande doit afficher `OpenFOAM-13`.

### 2. Cantera Python dans un environnement reproductible

La référence temporelle et le validateur utilisent l’interpréteur Python actif. Un environnement virtuel évite de dépendre d’une installation globale :

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'cantera>=3.2,<4' numpy pytest
python -c 'import cantera as ct; print("Cantera Python", ct.__version__)'
```

### 3. Cantera C++ depuis les sources

L’exemple ci-dessous reproduit l’installation C++ utilisée pour la validation. Les dépendances système sont installées explicitement, puis Cantera est construit avec Eigen système. `doxygen` est requis par la configuration de build actuelle.

```bash
sudo apt-get install -y build-essential gfortran libeigen3-dev libboost-dev doxygen
python3 -m pip install --user scons ninja
cd ..
git clone --depth 1 https://github.com/Cantera/cantera.git cantera-cxx
cd cantera-cxx
scons build -j2 system_eigen=y doxygen_docs=y sphinx_docs=n
scons install prefix="$HOME/.local/cantera"
```

Définir ensuite les variables utilisées par `Make/options` et par `Allrun` :

```bash
cd ../foampilot
export FOAM_BASHRC=/opt/openfoam13/etc/bashrc
source "$FOAM_BASHRC"
export CANTERA_ROOT="$HOME/.local/cantera"
export CANTERA_DATA="$CANTERA_ROOT/share/cantera/data"
export LD_LIBRARY_PATH="$CANTERA_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

Les contrôles suivants doivent tous réussir :

```bash
python -c 'import cantera as ct; print(ct.__version__)'
test -f "$CANTERA_ROOT/include/cantera/base/Solution.h"
test -f "$CANTERA_ROOT/lib/libcantera_shared.so"
test -f "$CANTERA_DATA/gri30.yaml"
ls -lh "$CANTERA_DATA/gri30.yaml"
```

Le fichier `gri30.yaml` est fourni dans les données Cantera installées. Il est utilisé par Python et par le bridge C++ via `canteraProperties`; `CANTERA_DATA` permet à Cantera de le résoudre sans dépendre du répertoire courant.

## Quick start de validation

Depuis la racine de `foampilot`, après les étapes d’installation ci-dessus :

```bash
. .venv/bin/activate
export FOAM_BASHRC=/opt/openfoam13/etc/bashrc
export CANTERA_ROOT="$HOME/.local/cantera"
export CANTERA_DATA="$CANTERA_ROOT/share/cantera/data"
export LD_LIBRARY_PATH="$CANTERA_ROOT/lib:${LD_LIBRARY_PATH:-}"
source "$FOAM_BASHRC"

cd cantera_openfoam13/validation/h2_autoignition
./Allrun
```

`Allrun` effectue exactement le pipeline suivant :

```text
wmake canteraFoam
    → blockMesh
    → canteraFoam
    → comparaison stricte du bridge avec Cantera
    → icoFoam
```

La référence gelée se trouve dans `validation/h2_autoignition/reference/cantera_reference.csv`. Les sorties et journaux générés sont placés dans `validation/h2_autoignition/results/` et `openfoam_case/canteraThermo.csv`; la référence n’est jamais écrasée.

Le validateur échoue avec un code non nul si OpenFOAM, Cantera C++ ou `canteraFoam` sont absents. Le mode `python validate.py --skip-openfoam` est réservé au contrôle explicite de la référence Cantera et ne constitue pas une validation OpenFOAM.

## Résultat attendu

La validation Foundation 13 vérifie un maillage de 1000 cellules, l’écriture de 1000 états thermochimiques, l’égalité à la référence d’équilibre `HP` avec tolérance numérique, puis un run `icoFoam` jusqu’à `0,02 s` sans `FOAM FATAL ERROR`. Cette validation démontre la chaîne logicielle et le bridge thermochimique. Elle ne revendique pas encore une validation scientifique d’un solveur de combustion transitoire couplé, car aucune source chimique n’est injectée dans les équations d’espèces ou d’énergie.

## Références

[1]: https://github.com/Cantera/cantera "Cantera — Chemical kinetics, thermodynamics, and transport tool suite"
[2]: https://cantera.org/stable/install.html "Cantera — Installation documentation"
[3]: https://openfoam.org/download/13-ubuntu/ "OpenFOAM Foundation — Download v13 for Ubuntu"
[4]: https://link.springer.com/article/10.1007/s10494-023-00449-8 "Zirwes et al. (2023), Assessment of Numerical Accuracy and Parallel Performance of OpenFOAM and its Reacting Flow Extension EBIdnsFoam"
