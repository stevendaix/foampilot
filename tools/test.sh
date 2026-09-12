#!/bin/bash

# Dossier racine pour la documentation
ROOT_DIR="."   # ou le chemin vers ton dossier

# Fichier de sortie
OUTPUT_FILE="file_list.txt"

# Supprime le fichier précédent s'il existe
rm -f "$OUTPUT_FILE"

# Parcours tous les fichiers et écrit le chemin relatif dans le fichier
find "$ROOT_DIR" -type f | sed "s|^$ROOT_DIR/||" > "$OUTPUT_FILE"

echo "Liste de tous les fichiers générée dans $OUTPUT_FILE"
