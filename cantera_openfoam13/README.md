# Couplage Cantera–OpenFOAM 13

Cette contribution ajoute à FoamPilot un adaptateur thermochimique **à échange de fichiers CSV**. OpenFOAM reste propriétaire du maillage et de l’avancement CFD; Cantera évalue la thermodynamique, le transport et la cinétique à partir de l’état de chaque cellule exporté par OpenFOAM. Le format CSV rend l’interface inspectable, testable et indépendante de l’ABI C++ de Cantera.

> Cette première intégration ne revendique pas un couplage implicite entièrement intégré dans `reactingFoam`. Elle fournit un couplage externe explicite, adapté à la validation et à l’orchestration batch de FoamPilot. Un adaptateur C++ in-process pourra être ajouté ultérieurement si la fréquence d’échange l’exige.

## Installation

Sur Ubuntu 24.04, la procédure Foundation est :

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common wget gnupg
sudo sh -c 'wget -qO- https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc'
sudo add-apt-repository -y 'http://dl.openfoam.org/ubuntu main dev'
sudo apt-get update
sudo apt-get install -y openfoam13
sudo pip3 install cantera
```

Chaque shell OpenFOAM doit ensuite charger l’environnement Foundation :

```bash
source /opt/openfoam13/etc/bashrc
foamVersion
python3 -c 'import cantera as ct; print(ct.__version__)'
```

## API d’échange

Le fichier d’entrée comporte les colonnes `cell`, `T`, `p` et `composition`. La composition suit la syntaxe Cantera, par exemple `"H2:2,O2:1,N2:3.76"`. Le fichier de sortie contient la température d’équilibre HP, la pression, la masse volumique, la capacité calorifique massique et la conductivité thermique.

```python
from foampilot.coupling.cantera_openfoam import CanteraOpenFOAMCoupler

coupler = CanteraOpenFOAMCoupler("gri30.yaml")
coupler.equilibrate_csv("openfoam_cells.csv", "cantera_cells.csv")
```

## Cas de validation

Le répertoire `validation/h2_autoignition` contient le pilote de validation homogène. Il calcule un délai d’auto-inflammation H2/air avec Cantera, puis exécute le cas OpenFOAM 13 lorsqu’un cas `chemFoam` et l’exécutable sont disponibles. Le critère est volontairement explicite : le run OpenFOAM doit retourner zéro, produire un journal sans `FOAM FATAL ERROR` et avancer jusqu’au temps final. La comparaison thermochimique de référence est produite par Cantera et stockée dans `results/cantera_reference.csv`.

Le benchmark de Zirwes et al. est utilisé comme motivation méthodologique pour séparer la validation logicielle de la validation quantitative des écoulements réactifs. Il n’est pas présenté comme reproduit intégralement par ce cas minimal.

## Références

[1]: https://github.com/Cantera/cantera "Cantera — Chemical kinetics, thermodynamics, and transport tool suite"
[2]: https://openfoam.org/download/13-ubuntu/ "OpenFOAM Foundation — Download v13 for Ubuntu"
[3]: https://link.springer.com/article/10.1007/s10494-023-00449-8 "Zirwes et al. (2023), Assessment of Numerical Accuracy and Parallel Performance of OpenFOAM and its Reacting Flow Extension EBIdnsFoam"
