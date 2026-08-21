# MakeHuman → STL/OBJ → zonage JOS-3

Cet exemple ajoute MakeHuman comme source de géométrie humaine réelle pour les cas de thermorégulation FoamPilot. Il récupère le maillage via le plugin socket MakeHuman, conserve le groupe de peau `body`, exclut les yeux, dents, cheveux et vêtements, puis produit un STL/OBJ global ainsi que 17 sous-surfaces correspondant au zonage JOS-3.

> Les 17 fichiers de zone sont des sous-surfaces destinées à définir les patches CFD. Ils ne doivent pas être utilisés comme 17 volumes indépendants. Le volume humain global doit être réparé, fermé et validé avant la soustraction du domaine fluide.

## Installation Ubuntu 24.04

Le PPA MakeHuman Community publié pour Ubuntu Noble est la voie recommandée :

```bash
cd examples/thermoregulation/makehuman
bash install_makehuman_ubuntu.sh
```

La procédure installe MakeHuman Community, ses données de morphologie et `xvfb` pour les tests sans écran physique. Elle active également les dépendances Python du pipeline dans l’environnement courant lorsque cela est possible.

Pour installer manuellement :

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:makehuman-official/makehuman-community
sudo apt-get update
sudo apt-get install -y makehuman-community python3-numpy python3-pip xvfb
python3 -m pip install --user numpy trimesh
```

MakeHuman Community 1.3.0 peut rencontrer des incompatibilités avec NumPy 2 dans certains environnements Python. Le symptôme est une erreur sur `fromstring`, `tostring` ou les noms d’uniformes OpenGL. Dans ce cas, utiliser l’environnement Python fourni par Ubuntu avec NumPy 1.26, ou appliquer le correctif local documenté dans `install_makehuman_ubuntu.sh` avec `--patch-numpy2`.

## Activer le socket MakeHuman

Créer le fichier utilisateur suivant :

```bash
mkdir -p "$HOME/makehuman/v1py3"
cat > "$HOME/makehuman/v1py3/socket.cfg" <<'JSON'
{
  "acceptConnections": true,
  "advanced": true,
  "host": "127.0.0.1",
  "port": 12345
}
JSON
```

Lancer MakeHuman dans un terminal séparé :

```bash
xvfb-run -a makehuman-community
```

Le serveur doit écouter sur `127.0.0.1:12345`. Pour une utilisation graphique, remplacer `xvfb-run -a makehuman-community` par `makehuman-community`.

## Export et zoning

Depuis la racine du dépôt :

```bash
python3 examples/thermoregulation/makehuman/export_makehuman_socket.py \
  --out examples/thermoregulation/makehuman/output

python3 examples/thermoregulation/makehuman/makehuman_stl_segmenter.py \
  examples/thermoregulation/makehuman/output/makehuman_body_only.stl \
  --out examples/thermoregulation/makehuman/output/jos3_zones \
  --export-global
```

Le pipeline crée notamment `makehuman_body_only.stl`, `makehuman_body_only.obj`, 17 fichiers `skin_<zone>.stl`, `manifest.json`, `zone_mapping.csv` et `quality_report.json`.

Le mapping CSV associe chaque triangle du STL source à un identifiant JOS-3. Ce mapping est un mapping géométrique de prétraitement. Après génération du maillage OpenFOAM, il doit être vérifié contre les patches réellement créés par snappyHexMesh ou Gmsh avant d’être injecté dans le driver CFD–JOS-3.

## Limites et prochaine étape

La classification de référence utilise des centroïdes dans un repère anthropométrique normalisé et détecte automatiquement l’axe vertical MakeHuman. Elle permet de tester la chaîne, mais elle ne remplace pas encore une classification par articulations. Pour les poses non neutres et les frontières mains/bras, la prochaine amélioration doit utiliser les repères du squelette MakeHuman ou des volumes de coupe construits autour des épaules, coudes, poignets, hanches, genoux et chevilles.

Le script est indépendant du protocole `externalCoupled` OpenFOAM 13. Le fichier `zone_mapping.csv` est conçu pour être consommé ensuite par le pilote FoamPilot/JOS-3 existant, qui doit agréger les températures et flux par zone au lieu d’utiliser un mapping fictif `face_id % 17`.
