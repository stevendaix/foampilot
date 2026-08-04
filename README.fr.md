<!-- Logo -->
<img src="foampilot/images/logo.png" alt="FoamPilot Logo" width="250">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/foampilot.svg)](https://pypi.org/project/foampilot/)
[![Docs](https://img.shields.io/badge/Docs-latest-blue.svg)](https://stevendaix.github.io/foampilot/)

# foampilot 🚀

🌍 **Langues :**  
[English](README.md) | [Français](README.fr.md) | [中文](README.zh.md)

**foampilot** est une plateforme Python conçue pour *orchestrer entièrement les simulations OpenFOAM* — de la définition du cas et du maillage à l’exécution,  
au post-traitement et à la génération de rapports.

Elle s’adresse aux ingénieurs et chercheurs souhaitant des flux de travail CFD **reproductibles, scriptables  
et maintenables**, sans avoir à éditer manuellement les dictionnaires OpenFOAM.

---

## Motivation

OpenFOAM est extrêmement puissant, mais gérer des simulations implique souvent :
- modification manuelle de plusieurs fichiers dictionnaires,
- duplication fragile des cas,
- scripts ad-hoc pour le post-traitement,
- reproductibilité limitée entre les études.

**foampilot** résout ces problèmes en plaçant Python au centre du workflow :  
les cas OpenFOAM deviennent des *artefacts générés*, et non des entrées maintenues manuellement.

---

## Fonctionnalités principales

- **Workflow Python-first**  
  Définissez les maillages, solveurs, conditions aux limites et contrôles directement en Python.

- **Génération automatique des cas OpenFOAM**  
  Génère les fichiers `system`, `constant` et `0/` de manière programmée, cohérente et reproductible.

- **Orchestration du maillage**  
  Support natif pour `blockMesh` et `snappyHexMesh`, avec une architecture extensible.

- **Contrôle des simulations**  
  Lancez et gérez les solveurs OpenFOAM directement depuis Python.

- **Post-traitement moderne**  
  Visualisation 3D avec PyVista, export automatique de figures et animations.

- **Rapports automatisés**  
  Génération de notes de calcul PDF (LaTeX) ou de tableaux de bord interactifs (Streamlit).

---

## Philosophie de conception

- Les dictionnaires OpenFOAM sont **générés**, jamais édités manuellement
- Reproductibilité et traçabilité privilégiées par rapport aux workflows GUI
- Configurations explicites et inspectables
- Conçu pour l’automatisation, les études paramétriques et les workflows d’ingénierie

---

## Ce que foampilot n’est *pas*

- Pas un solveur CFD  
- Pas un remplacement d’OpenFOAM  
- Pas un outil basé sur une interface graphique  
- Pas destiné à cacher les concepts OpenFOAM  

foampilot suppose une **familiarité de base avec OpenFOAM et la CFD**.

---

## Plateformes supportées

- **Linux** (natif)  
- **Windows via WSL2** (recommandé)  
- **macOS** (via les builds officiels OpenFOAM)

L’installation d’OpenFOAM et la configuration du système sont documentées séparément.

---

## Documentation

📘 Documentation complète, incluant les guides d’installation et l’utilisation détaillée :

**https://stevendaix.github.io/foampilot/fr/**

La documentation inclut :
- Guides d’installation OpenFOAM & WSL
- Structure et concepts du projet
- Maillage, contrôle des solveurs et post-traitement
- Génération de rapports et visualisation

---

## Statut du projet

⚠️ **Statut :** en développement / bêta

L’API publique peut évoluer.  
Vos retours, discussions et contributions sont les bienvenus.

---

## Licence

Ce projet est publié sous la **licence MIT**.
