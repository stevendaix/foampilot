# Constats de portage OpenFOAM 13

## Sources externes

[1] La page officielle OpenFOAM 13 indique que la version 13 réécrit les zones de maillage, introduit ou généralise les modèles `fvModels`, améliore les simulations multiphasiques et fournit de nouveaux modèles de rotor/propu­lseur. Source : https://openfoam.org/release/13/

[2] La procédure officielle Ubuntu installe OpenFOAM 13 via le paquet `openfoam13` et charge l’environnement avec `. /opt/openfoam13/etc/bashrc`. Source : https://openfoam.org/download/13-ubuntu/

## Constats techniques

Le dépôt `fronterapp/thesis-FloatingTurbine` annonce explicitement OpenFOAM v2012. Sa bibliothèque `floatingSixDoFRigidBodyMotion` recopie le cœur sixDoF historique, tandis que `floatingTurbinesFoam` s’appuie sur l’API `fvOptions`.

Dans OpenFOAM 13, l’API native sixDoF est distribuée sous `src/rigidBodyMotion/sixDoFRigidBodyMotion` et sa bibliothèque est `libsixDoFRigidBodyMotion.so`. Le cœur historique copié échoue à compiler avec OpenFOAM 13 à cause notamment de `dictionary::get`, `dictionary::getOrDefault`, `autoPtr::clone` et `Time::timeName()` incompatibles.

Une première stratégie de portage validée consiste à ne pas recopier le cœur sixDoF : le plugin `mooringLine` et `catenaryShape` se compilent contre la bibliothèque native OpenFOAM 13 avec `-lsixDoFRigidBodyMotion` et `-lfiniteVolume`. Cette compilation a produit `libfloatingSixDoFRigidBodyMotion.so` sans avertissement ni erreur.

Le code upstream `turbinesFoam` reste basé sur `fvOptions` et `cellSetOption`. OpenFOAM 13 ne fournit plus `cellSetOption` dans les chemins inspectés ; une conversion actuator-line complète devra donc remplacer cette base par `Foam::fv::fvModel`, adapter le constructeur `(name, modelType, mesh, dict)`, retirer les paramètres `fieldI`, fournir `addSupFields()` et implémenter les callbacks de topologie exigés par `fvModel`.

## Références

[1]: https://openfoam.org/release/13/ "OpenFOAM 13 Released"
[2]: https://openfoam.org/download/13-ubuntu/ "Download v13 | Ubuntu"
