# OpenFOAM on Windows – WSL Installation Guide

The **Windows Subsystem for Linux (WSL)** allows you to run a full Linux environment directly on Windows 10 and 11. This is currently the recommended method to run OpenFOAM reliably on Windows.

---

## Download Links

- Official OpenFOAM website: https://www.openfoam.com
- OpenFOAM v13: https://www.openfoam.com/download
- OpenFOAM-dev: https://www.openfoam.com/download/dev
- Running OpenFOAM on Windows (WSL): https://www.openfoam.com/download/windows
- Running OpenFOAM on macOS: https://www.openfoam.com/download/mac
- Compile from source: https://www.openfoam.com/download/source
- Package repository: https://dl.openfoam.org
- Version history: https://www.openfoam.com/releases

---

## Installing WSL

### 1. Open Windows Command Prompt as Administrator
- Click on the **Start Menu**, type `cmd`.
- Right-click on **Command Prompt** → *Run as administrator*.
- Confirm any authorization prompts.

### 2. Check if WSL is installed
Run the command:

```bash
wsl -l -v
````

* If no distributions are installed, proceed to the next step.
* Otherwise, check that Ubuntu is listed with version 2.

### 3. Install WSL with Ubuntu 22.04

Run:

```bash
wsl --install -d Ubuntu-22.04
```

* Follow the prompts to set up your Linux username and password.

### 4. Start WSL

* Open the **Start Menu**, search for **Ubuntu**, and launch it.
* Or type `wsl` in the Command Prompt.

---

## Installing OpenFOAM and ParaView

OpenFOAM and ParaView can be installed using the `apt` package manager. You will need superuser privileges (`sudo`) to run the following commands.

### 1. Add the OpenFOAM repository and public key

Run in the terminal:

```bash
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
```

**Note:**

* The public key uses `https://`.
* The repository URL uses `http://` because `https://` may not be supported, but security is ensured by the GPG key.

### 2. Update your package list

```bash
sudo apt update
```

### 3. Install OpenFOAM 13

```bash
sudo apt -y install openfoam13
```

OpenFOAM 13 and ParaView will be installed in `/opt`.

---

## User Configuration

To use OpenFOAM, follow these steps:

1. Open your `.bashrc` file in your home directory with an editor, for example:

```bash
gedit ~/.bashrc
```

2. Add the following line at the end of the file, then save and close:

```bash
. /opt/openfoam13/etc/bashrc
```

3. Open a new terminal window and test the installation:

```bash
foamRun -help
```

4. You should see the help message. Installation and configuration are now complete.

---

### Notes

* If you already have a similar line in `.bashrc` from a previous OpenFOAM version, comment it out with `#` or remove it.
* To apply changes immediately in the same terminal window after modifying `.bashrc`:

```bash
. $HOME/.bashrc
```

---

## Graphics and LaTeX Dependencies

### Install libraries for Gmsh and OpenGL

```bash
sudo apt install libglu1-mesa libgl1-mesa-glx libxrender1 libxext6
```

### Install TexLive for LaTeX

#### 1. Base installation

```bash
sudo apt-get install texlive-latex-base
```

#### 2. Recommended and extra fonts

To avoid errors when using `pdflatex` on LaTeX files with multiple fonts:

```bash
sudo apt-get install texlive-fonts-recommended
sudo apt-get install texlive-fonts-extra
```

#### 3. Additional LaTeX packages

```bash
sudo apt-get install texlive-latex-extra
```
