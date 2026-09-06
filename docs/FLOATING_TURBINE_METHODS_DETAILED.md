# Méthodes physiques et implémentation de la turbine flottante

## 1. Objet et périmètre

Ce document décrit la formulation théorique et l’implémentation logicielle des modèles intégrés à Foampilot pour une turbine flottante dans OpenFOAM 13. L’architecture combine un écoulement incompressible résolu sur un maillage eulérien, une représentation aérodynamique actuator-line des pales, un modèle de mouvement de corps rigide à six degrés de liberté et des efforts d’amarrage quasi statiques.

Le code est issu du dépôt [`thesis-FloatingTurbine`](https://github.com/fronterapp/thesis-FloatingTurbine) et a été refactorisé dans [`third_party/openfoam13`](../third_party/openfoam13). La bibliothèque actuator-line portée est `libturbinesFoam.so`, tandis que les restraints sixDoF sont regroupées dans `libfloatingSixDoFRigidBodyMotion.so`.

> **Statut important.** Le cas minimal séquentiel et MPI fonctionne avec OpenFOAM 13. Il s’agit d’une validation d’intégration et non d’une qualification scientifique de la turbine. Les coefficients doivent encore être confrontés à une référence expérimentale, BEM ou haute fidélité.

## 2. Équations de l’écoulement

Pour le cas incompressible, le solveur résout la conservation de la masse et de la quantité de mouvement :

$$
\\nabla\\cdot\\mathbf{U}=0,
$$

$$
\\frac{\\partial \\mathbf{U}}{\\partial t}
+\\nabla\\cdot(\\mathbf{U}\\otimes\\mathbf{U})
= -\\frac{1}{\\rho}\\nabla p
+\\nabla\\cdot(\\nu\\nabla\\mathbf{U})
+\\mathbf{f}_{ALM}.
$$

Dans le cas laminaire de validation, `physicalProperties` définit un fluide newtonien avec une viscosité cinématique `nu`. Le terme $\\mathbf{f}_{ALM}$ est une force volumique introduite par le modèle actuator-line. Dans un calcul turbulent, le terme visqueux effectif inclut la viscosité turbulente et la configuration `momentumTransport` sélectionne le modèle de fermeture.

La force ajoutée par l’ALM est exprimée par unité de masse dans certaines sorties historiques du module. La conversion en force volumique dépend de la convention du solveur et de la densité utilisée. Il faut donc vérifier les dimensions du champ source et la normalisation lorsque l’on change de solveur ou de modèle de densité.

## 3. Méthode actuator-line

### 3.1 Principe

L’Actuator Line Method (ALM) remplace la géométrie résolue de chaque pale par une ligne de points porteurs. À chaque point, la vitesse du fluide est interpolée depuis le champ CFD. La vitesse relative, l’angle d’attaque et les coefficients aérodynamiques sont ensuite calculés à partir de tables de profils. Les forces locales sont enfin projetées sur le maillage sous forme de force volumique régulière.

Cette approche évite de mailler la couche limite et la géométrie détaillée des pales, tout en conservant une dépendance locale au champ instationnaire. Elle est plus coûteuse et plus informative qu’un disque actuateur, mais moins coûteuse qu’un calcul blade-resolved.

La littérature montre que la précision dépend fortement de la largeur de projection $\\epsilon$, de la taille de maille $\\Delta_{grid}$ et de l’espacement des points d’actuator $\\Delta_{blade}$. Une règle pratique souvent utilisée est $\\epsilon \\gtrsim 2\\Delta_{grid}$, tandis que l’espacement des points doit rester de l’ordre de quelques tailles de maille [2].

### 3.2 Cinématique locale

Pour une turbine à axe horizontal, on définit un repère global $(x,y,z)$ et un repère local attaché à la rotation. Pour un point situé au rayon $r$, avec vitesse angulaire $\\Omega$, la composante tangentielle relative est approximativement :

$$
U_\\theta = \\Omega r - U_y\\cos\\theta + U_z\\sin\\theta.
$$

La composante axiale locale est $U_x$. La vitesse relative et l’angle d’écoulement valent :

$$
U_{rel}=\\sqrt{U_x^2+U_\\theta^2},
\\qquad
\\phi=\\operatorname{atan2}(U_x,U_\\theta).
$$

En notant $\\gamma$ la somme du vrillage local et du pas de pale, l’angle d’attaque est :

$$
\\alpha=\\phi-\\gamma.
$$

Le code calcule cette cinématique dans `actuatorLineElement::calculateInflowVelocity()` et `actuatorLineElement::calculateForce()`. La vitesse n’est pas prise directement dans la cellule la plus proche uniquement : le module peut échantillonner plusieurs points sur un cercle autour de l’élément, puis effectuer une moyenne afin de réduire la sensibilité aux fluctuations locales.

### 3.3 Forces de profil

Les tables de profil fournissent les coefficients de portance, traînée et moment en fonction de l’angle d’attaque, et éventuellement du nombre de Reynolds. Les forces par unité de longueur sont :

$$
 f_L=\\frac12\\rho U_{rel}^2 c C_L(\\alpha,Re),
\\qquad
 f_D=\\frac12\\rho U_{rel}^2 c C_D(\\alpha,Re).
$$

Ici $c$ est la corde locale. La force dans le repère global est obtenue par rotation depuis le repère normal/tangent de la section. Le code conserve aussi une force de moment lorsque le modèle de profil la demande.

Les profils sont lus par `profileData`. Les tables interpolées doivent couvrir la plage d’angles rencontrée. Une extrapolation insuffisamment contrôlée au-delà du décrochage peut produire des forces artificiellement élevées ou des discontinuités numériques.

### 3.4 Projection gaussienne

Une force discrète de pale ne peut pas être ajoutée à un seul centre de cellule sans produire une dépendance excessive à la position de la ligne. Le module distribue donc la force sur les cellules voisines avec un noyau gaussien isotrope :

$$
\\eta(\\mathbf{x})
= \\frac{1}{\\pi^{3/2}\\epsilon^3}
\\exp\\left[-\\left(\\frac{\\|\\mathbf{x}-\\mathbf{x}_0\\|}{\\epsilon}\\right)^2\\right].
$$

La contribution volumique est de la forme :

$$
\\mathbf{f}_{ALM}(\\mathbf{x})
= -\\mathbf{F}_{elem}\\,\\eta(\\mathbf{x}),
$$

avec le signe déterminé par la convention action/réaction. Dans `actuatorLineElement::applyForceField()`, la boucle sélectionne les cellules dans une sphère d’influence et ajoute la contribution lorsque la distance reste inférieure au rayon de projection.

Le code estime $\\epsilon$ à partir de la corde, de la traînée et de la taille locale de maille. Cette décision est importante : une largeur trop petite amplifie les oscillations et peut rendre le calcul instable ; une largeur trop grande lisse les tourbillons de bout et dégrade les charges locales. Les études ALM recommandent de vérifier le ratio $\\epsilon/\\Delta_{grid}$ et l’espacement axial ou radial des points avant toute comparaison physique [2] [3].

### 3.5 Effets de bout et de racine

Le module propose des corrections d’extrémité de type Glauert, Shen ou lifting-line selon la configuration. Elles modifient principalement la charge locale près du bout et de la racine. Une correction empirique ne doit pas être activée simultanément avec une résolution suffisante du tourbillon de bout sans étude de sensibilité : l’ALM peut déjà représenter une partie de la perte de charge par la dynamique du sillage.

Une pratique recommandée consiste à comparer au moins trois configurations : sans correction, correction de bout seule et correction combinée bout/racine. Les grandeurs à comparer sont le couple, la poussée, $C_P$, $C_T$ et la distribution de charge le long de la pale.

## 4. Décrochage dynamique de Leishman–Beddoes

### 4.1 Motivation

Une table statique $C_L(\\alpha)$ suppose que le profil réagit instantanément à l’angle d’attaque. Cette hypothèse devient fausse lorsque l’angle d’attaque varie rapidement, par exemple pendant un mouvement de tangage, de cavalement ou d’oscillation de plateforme. Le décrochage dynamique introduit un retard aérodynamique et une hystérésis entre montée et descente de l’incidence.

### 4.2 Structure du modèle

La famille Leishman–Beddoes est un modèle à variables d’état et fonctions indicielles. Sans reproduire toutes les constantes expérimentales, sa structure peut être résumée comme suit :

1. filtrage de l’incidence effective et séparation entre réponse attachée et réponse décrochée ;
2. évolution temporelle d’états représentant la circulation non stationnaire ;
3. apparition, convection et perte de portance associées au décollement ;
4. correction de traînée et de moment ;
5. reconstruction de $C_L$, $C_D$ et $C_M$ à l’instant courant.

Une variable d’état générique $x$ suit typiquement une relaxation exponentielle :

$$
 x^{n+1}=x^n e^{-\\Delta t/T}
+x_{eq}^{n+1}(1-e^{-\\Delta t/T}),
$$

où $T$ est une constante de temps adaptée au régime local. Les variantes `LeishmanBeddoes`, `LeishmanBeddoes3G`, `LeishmanBeddoesSGC` et `LeishmanBeddoesSD` implémentent des choix différents pour les états et les corrections.

### 4.3 Implémentation

La fabrique runtime `dynamicStallModel::New()` sélectionne la variante à partir du dictionnaire. `actuatorLineElement::calculateForce()` calcule l’incidence instantanée puis délègue au modèle dynamique lorsque `dynamicStall.active` est activé. Les profils fournissent l’état statique de référence ; les états dynamiques sont maintenus par élément actuator et doivent donc être cohérents avec `deltaT`.

Une attention particulière est nécessaire en MPI et lors d’un changement de maillage : les états appartiennent aux éléments physiques, tandis que les cellules CFD sont redistribuées. Il faut préserver l’identité et l’ordre des éléments lors de `mapMesh` et `distribute` si une simulation reprend après décomposition ou redistribution.

## 5. Mouvement de corps rigide à six degrés de liberté

### 5.1 Équations

Le flotteur est modélisé comme un corps rigide avec trois translations et trois rotations. La translation suit :

$$
 m\\frac{d\\mathbf{v}}{dt}=\\mathbf{F}_{hydro}+\\mathbf{F}_{aero}+\\mathbf{F}_{moor}+m\\mathbf{g}.
$$

La rotation suit l’équation d’Euler dans le repère corps :

$$
\\mathbf{I}\\frac{d\\boldsymbol{\\omega}}{dt}
+\\boldsymbol{\\omega}\\times(\\mathbf{I}\\boldsymbol{\\omega})
=\\mathbf{M}_{hydro}+\\mathbf{M}_{aero}+\\mathbf{M}_{moor}.
$$

La position et l’orientation sont intégrées par le solveur natif `sixDoFRigidBodyMotion` d’OpenFOAM 13. Le portage ne duplique donc pas le cœur sixDoF : il fournit des restraints spécialisées enregistrées dans la table runtime native.

### 5.2 Restraints

`constantLoad` applique une force et un moment prescrits. `mooringLine` calcule une force de rappel associée à une ligne d’amarrage. Les restraints sont évaluées à chaque pas et leur moment est obtenu à partir du bras de levier entre le centre de masse et le point d’application :

$$
\\mathbf{M}= (\\mathbf{x}_{app}-\\mathbf{x}_{CM})\\times\\mathbf{F}.
$$

Les contraintes de mouvement comme `axis` ou `line` réduisent l’espace des mouvements autorisés. Elles ne doivent pas être confondues avec une force de rappel : une contrainte impose un sous-espace cinématique, alors qu’une restraint fournit une contribution dynamique.

## 6. Ligne d’amarrage quasi statique

### 6.1 Modèle physique

Le modèle de ligne considère une ligne flexible soumise à son poids apparent, sa tension et sa géométrie de contact avec le fond. Dans l’approximation quasi statique, la ligne est supposée atteindre instantanément l’équilibre pour chaque position du flotteur.

Pour un segment de ligne, l’équilibre différentiel peut être écrit sous la forme :

$$
\\frac{d\\mathbf{T}}{ds}+\\mathbf{w}=\\mathbf{0},
$$

où $s$ est l’abscisse curviligne, $\\mathbf{T}$ la tension et $\\mathbf{w}$ le poids apparent par unité de longueur. La résolution fournit la tension au point d’attache et donc l’effort transmis au flotteur.

### 6.2 Implémentation `mooringLine`

La restraint portée lit l’ancrage, le point d’attache, la longueur, la masse linéique, le diamètre et le vecteur de gravité. `catenaryShape` calcule la géométrie et les composantes de tension. La bibliothèque est chargée avec :

```foam
libs ("libfloatingSixDoFRigidBodyMotion.so");
```

Puis la restraint est déclarée dans `dynamicMeshDict` :

```foam
mooring1
{
    sixDoFRigidBodyMotionRestraint mooringLine;
    anchor          (-100 0 -200);
    refAttachmentPt (-20 0 -14);
    massPerLength   108.63;
    lineLength      865.5;
    thickness       0.0766;
    gravityVector   (0 0 -9.8065);
}
```

Ce modèle ne représente pas la dynamique hydrodynamique complète d’une ligne, les effets de traînée distribuée, le contact détaillé avec le fond ou la rupture. Pour ces effets, il faudrait un couplage avec un solveur de lignes dynamique tel que MoorDyn.

## 7. Portage vers OpenFOAM 13

### 7.1 Changement `fvOption` vers `fvModel`

Les versions historiques utilisent `fvOptions` et `cellSetOption`. OpenFOAM 13 utilise des modèles enregistrés dans `fvModels`, avec des signatures field-based. Le portage suit la structure suivante :

| Ancienne API | API OpenFOAM 13 portée |
| --- | --- |
| `cellSetOption` | couche locale `cellSetOption.H` héritant de `fvModel` |
| `addSup(eqn, fieldI)` | `addSup(field, eqn)` |
| `selectionMode/cellSet` | `fvCellZone` et clé `cellZone` |
| `fvMesh::findCell` | `meshSearch::New(mesh).findCell` |
| `transportProperties` | `physicalProperties` pour le cas incompressible OF13 |
| table runtime `fvOption` | table runtime `fvModel` |

La couche de transition conserve temporairement des méthodes legacy afin de limiter la réécriture de l’algorithme aérodynamique. Les modèles doivent néanmoins être déclarés dans `constant/fvModels`, et leur bibliothèque doit être disponible dans `FOAM_USER_LIBBIN` ou chargée depuis `controlDict`.

### 7.2 Callbacks de maillage

Un modèle source doit rester cohérent lorsqu’un maillage bouge ou est redistribué. Les callbacks portés sont :

- `movePoints()` après déplacement des points ;
- `topoChange()` après modification topologique ;
- `mapMesh()` après mapping entre maillages ;
- `distribute()` après décomposition ou redistribution MPI.

La zone de cellules doit être mise à jour dans ces callbacks. Les états aérodynamiques attachés aux éléments de pale doivent, eux, rester associés à l’élément physique et non à un numéro de rang MPI.

## 8. Intégration Foampilot

La classe Python `FloatingTurbine` est déclarative. Elle valide les axes unitaires et les dimensions positives, puis génère :

```python
turbine.render_fv_models(cell_zone="rotor")
turbine.render_dynamic_mesh(...)
turbine.configure_solver(solver)
turbine.write(case_path, source_container="fvModels")
```

`configure_solver()` ajoute `libturbinesFoam.so` et, lorsque des lignes d’amarrage existent, `libfloatingSixDoFRigidBodyMotion.so`. Le code Python ne calcule pas les forces : il produit des dictionnaires reproductibles et vérifiables avant l’appel du solveur.

La séparation des responsabilités est volontaire :

| Couche | Responsabilité |
| --- | --- |
| Foampilot Python | validation d’entrées et génération des dictionnaires |
| `fvModel` actuator-line | couplage force/champ CFD |
| `actuatorLineElement` | cinématique, profil, décrochage et projection |
| `sixDoFRigidBodyMotion` | intégration du mouvement du flotteur |
| `mooringLine` | force et moment de rappel quasi statiques |
| OpenFOAM solver | discrétisation, pression, vitesse et pas de temps |

## 9. Parallélisme MPI

En MPI, chaque rang possède une portion du maillage. Les éléments de pale sont construits de manière cohérente sur les rangs, puis les réductions globales assurent une valeur commune pour les quantités nécessaires : vitesse interpolée, densité locale, largeur de projection et forces intégrées.

Les écritures de performance sont protégées par `Pstream::master()` afin d’éviter des fichiers concurrents. Les sorties doivent donc être comparées après reconstruction ou à partir des fichiers écrits par le rang maître.

Le test exécuté avec deux rangs a produit :

| Grandeur | $t=0,01$ s | $t=0,02$ s |
| --- | ---: | ---: |
| `Cp` | 0,380156337988 | 0,451833183431 |
| Coefficient de traînée rotor | 0,593972466956 | 0,667113188643 |
| Continuité globale cumulée | -3,455e-07 | -2,292e-07 |

La différence avec le cas séquentiel est inférieure à $3\\times10^{-7}$ sur `Cp` au second pas. Cette observation valide le chemin MPI du smoke test, mais ne remplace pas un test de scaling ou une validation sur une sélection de zone distribuée non triviale.

## 10. Guide de configuration

Un dictionnaire OF13 minimal ressemble à ceci :

```foam
 turbine
 {
     type axialFlowTurbineALSource;
     active on;
     axialFlowTurbineALSourceCoeffs
     {
         fieldNames         (U);
         selectionMode      cellZone;
         cellZone           all;
         cellSet             turbine; // compatibilité transitoire
         origin              (0 0 0);
         axis                (-1 0 0);
         verticalDirection  (0 0 1);
         freeStreamVelocity (10 0 0);
         tipSpeedRatio      6.0;
         rotorRadius        0.45;
         dynamicStall
         {
             active off;
         }
         blades
         {
             // blade1, blade2, blade3 et leurs elementData
         }
         profileData
         {
             cylinder
             {
                 data ((-180 0 1.1)(180 0 1.1));
             }
         }
     }
 }
```

Pour un cas réel, remplacer `cellZone all` par une zone physique construite par `topoSet` ou par le nouveau système de zones OpenFOAM 13. Vérifier que tous les points actuator se trouvent dans le domaine et que la largeur gaussienne est résolue par plusieurs cellules.

## 11. Procédure de validation recommandée

La validation doit être progressive. D’abord, lancer `blockMesh` et `checkMesh`. Ensuite, vérifier que le solveur affiche `Selecting finite volume model type axialFlowTurbineALSource` et les noms des pales. Après cela, exécuter un seul pas avec décrochage dynamique désactivé, puis activer les modèles un par un.

La seconde étape compare le calcul séquentiel et MPI sur le même maillage et le même `deltaT`. Les quantités à comparer sont le couple, la poussée, les forces par pale, `Cp`, `Ct`, les résidus de pression et l’erreur de continuité. La troisième étape raffine le maillage et diminue la largeur de projection en maintenant un ratio documenté $\\epsilon/\\Delta_{grid}$.

Enfin, une validation flottante doit coupler actuator-line, mouvement sixDoF et amarrages. Il faut alors vérifier séparément l’équilibre des forces, le signe des moments, la conservation de l’énergie et la sensibilité au pas de temps.

## 12. Limites et pièges fréquents

Une valeur de `Cp` non nulle ne prouve pas à elle seule que le modèle est physiquement correct. Elle montre que le modèle a été chargé, qu’une vitesse a été interpolée et que les coefficients de profil ont généré une force. Il faut encore vérifier les unités, les signes, le repère de rotation et la position des points.

Une largeur de projection trop grande peut donner un champ lisse et stable tout en sous-estimant les structures de sillage. Une largeur trop faible peut produire des oscillations et une forte dépendance au maillage. De même, augmenter le nombre de points de pale ne compense pas un maillage rotorique insuffisant.

La configuration `cellSet turbine` conservée dans le cas actuel est une compatibilité de transition avec le code historique. La cible de long terme est de supprimer cette dépendance et de faire lire exclusivement une sélection `cellZone` native par le modèle `fvModel`.

## Références

[1]: https://github.com/fronterapp/thesis-FloatingTurbine "Dépôt thesis-FloatingTurbine, implémentations physiques originales"

[2]: https://arxiv.org/html/2201.09368v1 "Liu et al., Evaluating the accuracy of the actuator line model against blade element momentum theory in uniform inflow"

[3]: https://wes.copernicus.org/articles/9/601/2024/ "Melani et al., An insight into the capability of the actuator line method to resolve tip vortices"

[4]: https://openfoam.org/version/13/ "OpenFOAM 13, présentation officielle et évolutions des mesh zones et modèles"

[5]: https://openfoam.org/download/13-ubuntu/ "Procédure officielle d’installation OpenFOAM 13 sous Ubuntu"
