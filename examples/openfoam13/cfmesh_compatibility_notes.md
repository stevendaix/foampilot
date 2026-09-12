# Vérification cfMesh pour Foundation 13

Le dépôt GitHub `wyldckat/cfMesh` consulté est un miroir non officiel de cfMesh version 1.0, avec un dernier commit visible daté de juin 2014. Son README indique une compilation via `Allwmake` dans un environnement OpenFOAM et l’utilisation de `cartesianMesh`, mais ne revendique pas de compatibilité avec OpenFOAM Foundation 13.

Le dépôt de référence propeller contient des fichiers `.fms` rotor/stator et des surfaces STL, mais l’environnement courant ne fournit ni `cartesianMesh` ni paquet cfMesh. Une compilation de ce miroir ancien contre Foundation 13 est donc une expérience de compatibilité à haut risque et ne doit pas être présentée comme garantie. Il faut identifier une source plus récente ou une branche explicitement adaptée à une version moderne d’OpenFOAM avant compilation.

Source consultée : https://github.com/wyldckat/cfMesh

Une branche `v2406` du projet GitLab Community `integration-cfmesh` a été consultée. Elle est indiquée comme **archivée et en lecture seule**. Elle fournit un `Allwmake` et une intégration cfMesh associée à l’écosystème OpenFOAM Community v2406, mais son README ne garantit pas la compatibilité avec la branche Foundation 13. Cette source est plus récente que le miroir v1.0, mais reste une base OpenCFD/Community à adapter et à compiler isolément.

Source consultée : https://develop.openfoam.com/Community/integration-cfmesh/-/tree/v2406

## Stratégie recommandée

La compilation directe de cfMesh v2406 contre Foundation 13 rencontre des changements structurels de conteneurs (`UList::setAddressableSize`) et de dictionnaires. Pour préserver le solver Foundation 13 sans entreprendre un fork complet de cfMesh, la voie la plus robuste est de compiler/exécuter cfMesh dans son environnement OpenFOAM Community v2406 natif, puis de transférer et contrôler le `constant/polyMesh` produit dans le cas Foundation 13. Le solveur, les modèles MRF/AMI et le runtime overset restent ainsi Foundation 13 ; seul le générateur de maillage est externe.
