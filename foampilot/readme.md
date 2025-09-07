foampilot 🚀

foampilot est une plateforme Python qui permet de piloter entièrement vos simulations OpenFOAM :
tout est défini dans des fichiers Python, depuis la création du maillage jusqu’au post-traitement et à la génération de rapports.

✨ Fonctionnalités principales

🔧 Configuration 100% Python : générez automatiquement vos fichiers OpenFOAM (system, constant, 0/…) sans édition manuelle.

📐 Gestion du maillage : support de blockMesh, snappyHexMesh et intégration possible d’autres mailleurs.

⚙️ Lancement automatisé : exécutez vos solveurs OpenFOAM directement depuis Python.

📊 Post-traitement moderne : visualisation 3D avec PyVista, export automatique en PNG/animations.

📝 Rapports automatiques : générez une note de calcul en PDF ou un dashboard interactif Streamlit pour présenter vos résultats.


🚀 Pourquoi foampilot ?

✅ Un seul langage pour tout contrôler : Python.

✅ Vérification plus simple et reproductible des cas.

✅ Gain de temps sur la préparation et l’analyse.

✅ Résultats présentés de manière moderne et professionnelle.


📌 Exemple rapide

from foampilot import Case

# Créer un cas simple avec blockMesh et turbulence k-epsilon
case = Case("demo_case")
case.mesh.blockmesh(resolution=(20,20,20))
case.boundary.add_velocity_inlet("inlet", value=(1,0,0))
case.boundary.add_pressure_outlet("outlet")

# Lancer la simulation
case.run("simpleFoam")

# Post-traitement
case.post.show_field("U", mesh=True)
case.report.to_pdf("rapport.pdf")

