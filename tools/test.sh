#!/bin/bash
cd docs

# Fichiers français
for file in fr/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        mv "$file" "${filename%.md}.fr.md"
        echo "Renommé: $file → ${filename%.md}.fr.md"
    fi
done

# Fichiers chinois
for file in zh/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        mv "$file" "${filename%.md}.zh.md"
        echo "Renommé: $file → ${filename%.md}.zh.md"
    fi
done

# Supprimer les dossiers vides
rmdir fr zh 2>/dev/null || true

echo "✅ Conversion terminée !"