# OpenFOAM sur Windows – Guide d'installation WSL

Le sous-système Windows pour Linux (WSL) vous permet d’exécuter des applications Linux directement sur Windows 10 et 11.

---

## Liens de téléchargement

- [Site Web officiel d'OpenFOAM](#)
- [OpenFOAM v13](#)
- [OpenFOAM-dev](#)
- [Exécuter OpenFOAM sous Windows](#)
- [Exécuter OpenFOAM sur macOS](#)
- [Compiler à partir de la source](#)
- [Dépôt de paquets](#)
- [Historique des versions](#)

---

## Installation de WSL

### 1. Ouvrez l'invite de commande Windows en tant qu'administrateur
- Cliquez sur le menu Démarrer, tapez « cmd ».
- Faites un clic droit sur « Invite de commandes » et sélectionnez « Exécuter en tant qu'administrateur ».
- Confirmez toutes les invites d’autorisation.

### 2. Vérifiez si WSL est installé
Exécutez la commande :
```bash
wsl -l -v
```
- Si aucune distribution n’est installée, passez à l’étape suivante.
- Sinon, vérifiez qu'Ubuntu est répertorié avec la version 2.

### 3. Installer WSL avec Ubuntu 22.04
Courir:
```bash
wsl --install -d Ubuntu-22.04
```
- Suivez les instructions pour définir le nom d'utilisateur et le mot de passe.

### 4. Démarrez WSL
- Ouvrez le menu Démarrer, recherchez « Ubuntu » et lancez-le.
- Ou tapez « wsl » dans l'invite de commande.

---

## Installation d'OpenFOAM et ParaView

OpenFOAM et ParaView s'installent facilement grâce au gestionnaire de paquets « apt ». Vous devrez saisir votre mot de passe de superutilisateur pour exécuter les commandes suivantes avec « sudo » :

### 1. Ajoutez le référentiel OpenFOAM et la clé publique
Exécutez ces commandes dans le terminal :
```bash
sudo sh -c "wget ​​-O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
```
Remarque : utilisez « https:// » pour la clé publique. L'URL du référentiel utilise « http:// » car « https:// » n'est peut-être pas pris en charge, mais la sécurité est assurée par la clé.

### 2. Mettez à jour votre liste de colis
```bash
mise à jour sudo apt
```

### 3. Installez OpenFOAM 13
```bash
sudo apt -y install openfoam13
```
OpenFOAM 13 et ParaView seront installés dans le répertoire `/opt`.

---

## Configuration utilisateur

Pour utiliser OpenFOAM, suivez ces étapes :

1. Ouvrez votre fichier `.bashrc` dans votre répertoire personnel avec un éditeur, par exemple :
   ```bash
   gedit ~/.bashrc
   ```

2. Ajoutez cette ligne à la fin du fichier, puis enregistrez et fermez :
   ```bash
   . /opt/openfoam13/etc/bashrc
   ```

3. Ouvrez une nouvelle fenêtre de terminal et testez l'installation en exécutant :
   ```bash
   foamRun -aide
   ```

4. Vous devriez voir le message d'aide. L'installation et la configuration sont maintenant terminées.

---

### Remarques
- Si vous avez déjà une ligne similaire dans `.bashrc` (d'une ancienne version d'OpenFOAM), commentez-la en ajoutant un `#` au début de la ligne ou supprimez-la.
- Pour appliquer les modifications immédiatement dans la même fenêtre de terminal après avoir modifié `.bashrc`, exécutez :
  ```bash
  . $HOME/.bashrc
  ```

installer ca pour gmsh sudo apt install libglu1-mesa libgl1-mesa-glx libxrender1 libxext6

Install the TexLive base
sudo apt-get install texlive-latex-base
Also install the recommended and extra fonts to avoid running into the error [1], when trying to use pdflatex on latex files with more fonts.
sudo apt-get install texlive-fonts-recommended
sudo apt-get install texlive-fonts-extra
Install the extra packages,
sudo apt-get install texlive-latex-extra
---

*Dernière mise à jour : juillet 2025*