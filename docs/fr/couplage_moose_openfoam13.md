# Couplage MOOSE–OpenFOAM 13 sans preCICE

## Décision d’architecture

Foampilot utilise ici le mécanisme **`externalCoupledTemperature` fourni par OpenFOAM 13**. OpenFOAM écrit les températures, flux et coefficients de transfert de la frontière dans un fichier texte, retire `OpenFOAM.lock`, puis attend que le participant externe écrive les valeurs de la condition mixte dans le fichier `.in` et recrée le verrou. Le participant MOOSE peut donc être implémenté avec `MultiApp`/`Transfers` ou avec un problème externe dérivé de `ExternalProblem`, sans introduire de bibliothèque de couplage supplémentaire.

Cette solution est adaptée à un couplage explicite et séquentiel de validation thermique. Elle ne fournit pas, à elle seule, les fonctions avancées d’un couplage partitionné distribué : interpolation géométrique générale, couplage implicite avec accélération ou équilibrage MPI entre solveurs. Ces fonctions justifieraient une bibliothèque spécialisée dans un second temps.

## Installation OpenFOAM 13 sous Ubuntu 24.04

Les commandes officielles sont les suivantes :

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common ca-certificates wget gnupg
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo rm -f /etc/apt/sources.list.d/*dl_openfoam_org*list
sudo add-apt-repository "http://dl.openfoam.org/ubuntu main dev"
sudo apt-get update
sudo apt-get install -y openfoam13
printf '\n. /opt/openfoam13/etc/bashrc\n' >> ~/.bashrc
source /opt/openfoam13/etc/bashrc
foamRun -help
```

Le paquet fournit également les bibliothèques et utilitaires nécessaires au cas de validation, notamment `blockMesh` et ParaView.

## Installation MOOSE

La méthode recommandée par MOOSE pour Linux consiste à installer Miniforge dans le répertoire utilisateur, ajouter le canal public INL, puis créer un environnement Conda. La variante précompilée convient à l’exécution des applications MOOSE ; pour développer une application MOOSE personnalisée, utiliser `moose-dev` et compiler le dépôt MOOSE.

```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge
export PATH="$HOME/miniforge/bin:$PATH"
conda config --add channels https://conda.software.inl.gov/public
mamba create -n moose moose
conda activate moose
moose --help
```

Pour un développement MOOSE depuis les sources :

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/idaholab/moose.git
cd moose
git checkout master
conda activate moose
cd test
make -j6
./run_tests -j6
```

## Utilisation depuis Foampilot

Le module `foampilot.coupling.ExternalCoupledTemperature` lit le fichier `comms/temperature.out`, décode les enregistrements par patch et écrit le fichier `comms/temperature.in`. Le verrouillage est atomique côté fichier temporaire : les données sont d’abord écrites dans un fichier `.tmp`, puis renommées avant la recréation de `OpenFOAM.lock`.

```python
from foampilot.coupling import ExternalCoupledTemperature

coupling = ExternalCoupledTemperature(
    comms_dir="comms",
    file_name="temperature",
    wait_interval=0.1,
    timeout=120.0,
)
records = coupling.wait_for_openfoam()
# Utiliser records dans le transfert MOOSE, puis renvoyer une condition mixte.
coupling.send_temperature_mixed_values(
    [(record.temperature, 0.0, 1.0) for record in records]
)
```

## Vérifications

Les tests unitaires vérifient le parsing de plusieurs patches, la création du fichier `.in` et du verrou, ainsi que l’expiration contrôlée d’un échange absent. Le cas de validation doit en outre vérifier que la température imposée côté MOOSE et le flux lu côté OpenFOAM convergent vers la solution analytique d’une conduction plane 1D.

## Références

[1]: https://mooseframework.inl.gov/getting_started/installation/index.html "Installing MOOSE"
[2]: https://mooseframework.inl.gov/getting_started/installation/conda.html "Conda MOOSE Environment"
[3]: https://mooseframework.inl.gov/syntax/MultiApps/index.html "MOOSE MultiApp System"
[4]: https://mooseframework.inl.gov/source/problems/ExternalProblem.html "MOOSE ExternalProblem"
[5]: https://cpp.openfoam.org/v13/classFoam_1_1externalCoupledTemperatureMixedFvPatchScalarField.html "OpenFOAM 13 externalCoupledTemperature"
[6]: https://openfoam.org/download/13-ubuntu/ "OpenFOAM 13 for Ubuntu"
