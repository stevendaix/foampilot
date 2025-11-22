#!/usr/bin/env bash
PYTHON_VERSION="3.11"

# Détecte le dossier contenant pyproject.toml
if [ -f "./pyproject.toml" ]; then
    PROJECT_DIR="$(pwd)"
elif [ -f "./foampilot/pyproject.toml" ]; then
    PROJECT_DIR="$(pwd)/foampilot"
else
    echo "❌ Aucun pyproject.toml trouvé"
    exit 1
fi

MODULE_NAME="foampilot"

docker run --rm -it \
    -v "$PROJECT_DIR":/app \
    -w /app \
    python:${PYTHON_VERSION} bash -c "
        apt-get update && apt-get install -y \
            # OpenGL / Mesa (pour GMSH et PyVista)
            libglu1-mesa \
            libgl1-mesa-glx \
            libxrender1 \
            libxext6 \
            libsm6 \
            libice6 \
            libglu1-mesa-dev \
            # LaTeX (pour pylatex)
            texlive-latex-base \
            texlive-fonts-recommended \
            texlive-fonts-extra \
            texlive-latex-extra \
            # Autres dépendances utiles
            git \
            wget
        pip install --upgrade pip
        rm -rf build/  # Nettoie le répertoire build s'il existe
        pip install .
        python -c 'import foampilot; print(\"✅ Module OK\")'
"
