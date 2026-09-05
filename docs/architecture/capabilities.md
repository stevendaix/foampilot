# Capabilities

Une capability fournit une fonction technique réutilisable et ne doit pas dépendre d’un workflow métier. Elle ne doit pas demander si son appelant est médical, marin ou urbain.

## Classification initiale

| Capability cible | Éléments actuels à cartographier | Règle de migration |
|---|---|---|
| Geometry | `foampilot/src/foampilot/geometry/`, CAD, surfaces, topology, VMTK | Conserver les primitives génériques ; extraire les étapes patient dans les workflows médicaux |
| Meshing | `foampilot/src/foampilot/meshing/`, Gmsh, blockMesh, snappyHexMesh et qualité | Les générateurs et contrôles de maillage restent réutilisables |
| Case | classes de structure de cas et chemins | Responsable de l’arborescence, fichiers et validation structurelle |
| Dictionaries | writers et modèles de dictionnaires | Ne pas dépendre d’un workflow particulier |
| Boundaries | gestion des patches et conditions limites | Recevoir des données explicites, sans import métier implicite |
| Postprocessing | analyse de champs, métriques et export | Séparer les métriques génériques des rapports médicaux ou urbains |
| Reporting | rapports, validation de contrats et synthèses | Produire des résultats traçables et indépendants du domaine |
| Execution | lancement de commandes et gestion des environnements | Déléguer la version OpenFOAM au backend |

## Capability et données

Une capability peut consommer une géométrie, un maillage, un dictionnaire ou un champ, mais ne doit pas importer directement un exemple ou un workflow. Les fixtures scientifiques restent dans `examples/`, `tutorials/` ou `validation/` selon leur rôle.

## Exemple VMTK

Les opérations de traitement de surface, extraction de centerline, extraction de sections, rééchantillonnage, topologie et conversion de maillage relèvent de la capability Geometry/VMTK. La reconstruction d’une aorte particulière, le prétraitement patient et la définition d’une campagne de CoA relèvent du workflow médical.

Cette séparation permet à un workflow marin ou urbain d’utiliser des primitives de géométrie ou de traitement de surface sans dépendre du paquet médical.
