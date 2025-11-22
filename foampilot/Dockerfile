FROM python:3.11

WORKDIR /app

# Installe les dépendances système
RUN apt-get update && apt-get install -y \
    # OpenGL / Mesa (pour GMSH et PyVista)
    libglu1-mesa \
    libgl1 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libice6 \
    libglu1-mesa-dev \
    libgl1-mesa-dri \
    # GMSH dépendances supplémentaires
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-dri2-0 \
    libxcb-dri3-0 \
    libxcb-present0 \
    libxcb-sync1 \
    libxshmfence1 \
    # LaTeX (pour pylatex)
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    # Autres dépendances utiles
    git \
    wget \
    && ldconfig

# Copie le projet
COPY . .

# Installe les dépendances Python
RUN pip install --upgrade pip && \
    pip install .

# Teste l'import
RUN python -c 'import foampilot; print("✅ Module OK")'
