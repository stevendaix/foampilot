# 📚 Documentation Théorique pour la Mécanique des Fluides

Ce document fournit un aperçu théorique des principes de mécanique des fluides mis en œuvre dans la classe Python `FluidMechanics`. Il vise à servir de guide complet pour les utilisateurs et les développeurs cherchant à comprendre la physique et les modèles mathématiques sous-jacents utilisés dans la bibliothèque pour les applications de Dynamique des Fluides Computationnelle (CFD).

## 1. Introduction à la Mécanique des Fluides pour la CFD

La mécanique des fluides est la branche de la physique qui s'intéresse à la mécanique des fluides (liquides, gaz et plasmas) et aux forces qui s'exercent sur eux. Elle a un large éventail d'applications, notamment l'aéronautique, le génie civil, la météorologie et le génie biomédical. Dans le contexte de la Dynamique des Fluides Computationnelle (CFD), les principes de la mécanique des fluides sont discrétisés et résolus numériquement pour simuler les phénomènes d'écoulement de fluide. La classe `FluidMechanics` encapsule plusieurs concepts fondamentaux et corrélations empiriques cruciaux pour la mise en place et l'analyse des simulations CFD, particulièrement concernant la génération de maillage et les conditions aux limites.

Les simulations CFD reposent fortement sur la compréhension du comportement des fluides à diverses échelles et conditions. Cela inclut la caractérisation du régime d'écoulement, la prédiction des pertes d'énergie et l'analyse des mécanismes de transfert de chaleur. La classe `FluidMechanics` fournit des outils pour quantifier ces aspects, ce qui facilite la préparation des entrées de simulation et l'interprétation des résultats pour les ingénieurs et les chercheurs. La sélection appropriée des propriétés du fluide, des paramètres d'écoulement et des nombres sans dimension est primordiale pour des calculs CFD précis et stables.

## 2. Nombres Sans Dimension en Mécanique des Fluides

Les nombres sans dimension sont cruciaux en mécanique des fluides car ils permettent la mise à l'échelle des phénomènes physiques et donnent un aperçu de l'importance relative des différentes forces agissant sur un fluide. Ils sont particulièrement utiles en CFD pour comparer différents scénarios d'écoulement et valider les résultats de simulation par rapport à des données expérimentales ou des solutions analytiques. La classe `FluidMechanics` calcule plusieurs nombres sans dimension clés, chacun fournissant des informations uniques sur l'écoulement.

### 2.1. Nombre de Reynolds (Re)

Le nombre de Reynolds est l'une des grandeurs sans dimension les plus importantes en dynamique des fluides, utilisée pour prédire les motifs d'écoulement dans différentes situations d'écoulement de fluide. Il est défini comme le rapport des forces d'inertie aux forces visqueuses et est donné par la formule :

$$Re = \frac{\rho v L}{\mu}$$

Où :
* $\rho$ est la masse volumique du fluide (kg/m³)
* $v$ est la vitesse d'écoulement caractéristique (m/s)
* $L$ est la dimension linéaire caractéristique (m)
* $\mu$ est la viscosité dynamique du fluide (Pa·s)

Pour les écoulements internes, comme l'écoulement dans un tuyau, la dimension linéaire caractéristique est typiquement le diamètre du tuyau. Pour les écoulements externes, il peut s'agir de la longueur d'une plaque ou du diamètre d'un cylindre. Le nombre de Reynolds aide à déterminer si l'écoulement est laminaire, transitoire ou turbulent. Généralement, pour l'écoulement dans un tuyau, $Re < 2300$ indique un écoulement laminaire, $2300 \le Re \le 4000$ indique un écoulement transitoire, et $Re > 4000$ indique un écoulement turbulent [1].

### 2.2. Nombre de Prandtl (Pr)

Le nombre de Prandtl est un nombre sans dimension qui approxime le rapport de la diffusivité de quantité de mouvement (viscosité cinématique) à la diffusivité thermique. Il est utilisé pour caractériser l'épaisseur relative des couches limites hydrodynamiques et thermiques. Il est défini comme :

$$Pr = \frac{\mu c_p}{k}$$

Où :
* $\mu$ est la viscosité dynamique (Pa·s)
* $c_p$ est la capacité thermique massique à pression constante (J/(kg·K))
* $k$ est la conductivité thermique (W/(m·K))

Pour les gaz, Pr est généralement d'environ 0,7-1,0, ce qui indique que la quantité de mouvement et la chaleur diffusent à des vitesses similaires. Pour les liquides, Pr peut varier considérablement. Par exemple, l'eau à température ambiante a un Pr d'environ 7, ce qui signifie que la quantité de mouvement diffuse beaucoup plus rapidement que la chaleur [2].

### 2.3. Nombre de Nusselt (Nu)

Le nombre de Nusselt est le rapport entre le transfert de chaleur par convection et par conduction à travers une limite. C'est un coefficient de transfert de chaleur sans dimension qui quantifie l'amélioration du transfert de chaleur d'une surface due à la convection par rapport à la conduction à travers la couche de fluide. Il est défini comme :

$$Nu = \frac{h L}{k}$$

Où :
* $h$ est le coefficient de transfert de chaleur par convection (W/(m²·K))
* $L$ est la longueur caractéristique (m)
* $k$ est la conductivité thermique du fluide (W/(m·K))

Un nombre de Nusselt plus élevé indique un transfert de chaleur par convection plus efficace. Pour la conduction pure, $Nu = 1$. Des corrélations empiriques sont souvent utilisées pour déterminer le nombre de Nusselt pour diverses géométries et conditions d'écoulement [3].

### 2.4. Nombre de Grashof (Gr)

Le nombre de Grashof est un nombre sans dimension en dynamique des fluides et en transfert de chaleur qui approxime le rapport de la force de flottabilité à la force visqueuse agissant sur un fluide. Il est principalement utilisé dans les problèmes de convection naturelle, où le mouvement du fluide est entraîné par des différences de densité dues à des variations de température. Il est défini comme :

$$Gr = \frac{g \beta \Delta T L^3}{\nu^2}$$

Où :
* $g$ est l'accélération due à la gravité (m/s²)
* $\beta$ est le coefficient de dilatation thermique (1/K)
* $\Delta T$ est la différence de température (K)
* $L$ est la longueur caractéristique (m)
* $\nu$ est la viscosité cinématique (m²/s)

Le nombre de Grashof joue un rôle similaire dans la convection naturelle à celui du nombre de Reynolds dans la convection forcée, indiquant la transition d'un écoulement laminaire à turbulent dans les écoulements entraînés par la flottabilité [4].

### 2.5. Nombre de Rayleigh (Ra)

Le nombre de Rayleigh est un nombre sans dimension associé à l'écoulement entraîné par la flottabilité (convection naturelle). Lorsque le nombre de Rayleigh est inférieur à une valeur critique pour un fluide, le transfert de chaleur se fait principalement par conduction ; lorsqu'il dépasse la valeur critique, le transfert de chaleur se fait principalement par convection. Il est défini comme le produit du nombre de Grashof et du nombre de Prandtl :

$$Ra = Gr \cdot Pr = \frac{g \beta \Delta T L^3}{\nu \alpha}$$

Où :
* $\alpha$ est la diffusivité thermique (m²/s)

Le nombre de Rayleigh critique varie en fonction de la géométrie et des conditions aux limites. Par exemple, pour une couche de fluide horizontale chauffée par le bas, la convection commence généralement lorsque $Ra > 1708$ [5].

### 2.6. Nombre de Peclet (Pe)

Le nombre de Peclet est un nombre sans dimension pertinent dans l'étude des phénomènes de transport dans les écoulements de fluide. Il est défini comme le rapport du taux d'advection d'une quantité physique par l'écoulement au taux de diffusion de la même quantité entraînée par un gradient approprié. Il est donné par :

$$Pe = Re \cdot Pr$$

Où :
* $Re$ est le nombre de Reynolds
* $Pr$ est le nombre de Prandtl

Alternativement, il peut être exprimé comme :

$$Pe = \frac{v L}{\alpha}$$

Où :
* $v$ est la vitesse d'écoulement (m/s)
* $L$ est la longueur caractéristique (m)
* $\alpha$ est la diffusivité thermique (m²/s)

Un grand nombre de Peclet indique que l'advection domine la diffusion, tandis qu'un petit nombre de Peclet suggère que la diffusion est plus significative. Ce nombre est particulièrement important dans les problèmes de transfert de chaleur et de masse [6].

## 3. Calculs de Couche Limite

Les couches limites sont de minces couches de fluide adjacentes aux surfaces solides où les effets visqueux sont significatifs. Comprendre et modéliser avec précision les couches limites est essentiel en CFD, en particulier pour prédire la traînée, la portance et le transfert de chaleur. La classe `FluidMechanics` fournit des outils pour estimer les caractéristiques de la couche limite, qui sont essentielles pour une génération de maillage appropriée dans les simulations CFD.

### 3.1. Valeur y+

La valeur y+ (prononcée "y-plus") est une distance sans dimension par rapport à la paroi, normalisée par l'échelle de longueur visqueuse. C'est un paramètre crucial dans la modélisation de la turbulence, particulièrement pour les écoulements limités par une paroi. Elle est définie comme :

$$y^+ = \frac{u_\tau y}{\nu}$$

Où :
* $u_\tau$ est la vitesse de frottement (m/s), définie comme $\sqrt{\tau_w / \rho}$
* $y$ est la distance physique par rapport à la paroi (m)
* $\nu$ est la viscosité cinématique (m²/s)
* $\tau_w$ est la contrainte de cisaillement pariétale (Pa)
* $\rho$ est la masse volumique du fluide (kg/m³)

Pour de nombreux modèles de turbulence, le centre de la première cellule à partir de la paroi doit être placé dans une plage de y+ spécifique [7].

### 3.2. Épaisseur de la Couche Limite Turbulente

L'épaisseur de la couche limite ($\delta$) est typiquement définie comme la distance par rapport à la paroi où la vitesse du fluide atteint 99 % de la vitesse du courant libre. Pour les couches limites turbulentes sur une plaque plate, une corrélation courante est :

$$\delta \approx 0.37 L Re_L^{-1/5}$$

Où :
* $L$ est la longueur caractéristique (m)
* $Re_L$ est le nombre de Reynolds basé sur la longueur caractéristique

Cette corrélation est une approximation [8].

### 3.3. Nombre de Cellules de Couche Limite pour le Dimensionnement du Maillage

En CFD, la résolution précise de la couche limite nécessite un nombre suffisant de mailles. La classe `FluidMechanics` estime le nombre de couches nécessaires pour atteindre une taille de cellule cible au bord de la couche limite, étant donné un rapport d'expansion (généralement entre 1,1 et 1,3) [9].

## 4. Calculs de Perte de Pression

La perte de pression dans les systèmes d'écoulement de fluide est un paramètre essentiel dans la conception technique.

### 4.1. Équation de Darcy-Weisbach

Elle relie la perte de pression ($\Delta P$) due au frottement le long d'une longueur donnée de tuyau à la vitesse moyenne du fluide.

$$\Delta P = f \frac{L}{D} \frac{\rho v^2}{2}$$

Où :
* $f$ est le facteur de frottement de Darcy (sans dimension)
* $L$ est la longueur du tuyau (m)
* $D$ est le diamètre intérieur du tuyau (m)
* $\rho$ est la masse volumique du fluide (kg/m³)
* $v$ est la vitesse d'écoulement moyenne (m/s)

### 4.2. Facteur de Frottement ($f$)

Le facteur de frottement dépend du régime d'écoulement et de la rugosité du tuyau.

* **Pour l'écoulement laminaire** ($Re < 2300$) :
    $$f = \frac{64}{Re}$$

* **Pour l'écoulement turbulent** ($Re \ge 2300$) :
    La classe utilise une approximation explicite basée sur l'équation de Colebrook-White, comme l'équation de Swamee-Jain, valable pour $Re > 4000$ :

    $$f = \frac{0.25}{\left[\log_{10}\left(\frac{\epsilon}{3.7D} + \frac{5.74}{Re^{0.9}}\right)\right]^2}$$
    Où $\epsilon$ est la rugosité absolue [10].

## 5. Calculs de Transfert de Chaleur

Le transfert de chaleur est un aspect fondamental des applications d'écoulement de fluide.

### 5.1. Coefficient de Transfert de Chaleur par Convection

Le taux de transfert de chaleur par convection ($Q$) est donné par la Loi de Refroidissement de Newton :

$$Q = h A \Delta T$$

Où $h$ est le coefficient de transfert de chaleur par convection, souvent déterminé par des corrélations impliquant les nombres de Nusselt, de Reynolds et de Prandtl.

#### 5.1.1. Plaque Plate (Écoulement Externe)

* **Écoulement Laminaire** ($Re_L < 5 \times 10^5$) :
    $$Nu_L = 0.664 Re_L^{0.5} Pr^{1/3}$$

* **Écoulement Turbulent** ($Re_L \ge 5 \times 10^5$) :
    $$Nu_L = 0.0296 Re_L^{0.8} Pr^{1/3}$$

Le coefficient $h$ est ensuite calculé par $h = Nu_L \frac{k}{L}$ [11].

#### 5.1.2. Cylindre (Écoulement Externe)

L'équation de Churchill-Bernstein donne le nombre de Nusselt moyen pour l'écoulement normal à un cylindre circulaire ($Re_D Pr > 0.2$) :

$$Nu_D = 0.3 + \frac{0.62 Re_D^{0.5} Pr^{1/3}}{\left[1 + (0.4/Pr)^{2/3}\right]^{0.25}} \left[1 + \left(\frac{Re_D}{282000}\right)^{0.5}\right]^{0.5}$$

Le coefficient $h$ est calculé par $h = Nu_D \frac{k}{D}$ [12].

### 5.2. Coefficient de Dilatation Thermique ($\beta$)

Quantifie le changement de masse volumique d'un fluide avec la température. Pour les gaz parfaits : $\beta = 1/T$ [13].

## 6. Propriétés des Fluides et Régimes d'Écoulement

### 6.1. Propriétés Fondamentales des Fluides

La classe utilise la librairie `pyfluids` pour récupérer des propriétés (masse volumique $\rho$, viscosité dynamique $\mu$, conductivité thermique $k$, chaleur spécifique $c_p$, volume spécifique $v$) à une température et pression données [14].

### 6.2. Détermination du Régime d'Écoulement

La classification (laminaire, transitoire, ou turbulent) est basée sur le nombre de Reynolds, essentielle pour choisir les modèles CFD appropriés [15].

### 6.3. Vitesse Critique ($v_c$)

Vitesse de transition, calculée pour un nombre de Reynolds critique ($Re_{crit} \approx 2300$) :

$$v_c = \frac{Re_{crit} \mu}{\rho D}$$ [16].

## 7. Validation des Entrées et Gestion des Erreurs

La classe utilise des méthodes de validation pour garantir la fiabilité et la cohérence physique des entrées :

### 7.1. Validation de Quantité Positive

Vérifie que des paramètres tels que la pression, la vitesse et la longueur sont $> 0$.

### 7.2. Validation de Quantité Non Nulle

Vérifie que les propriétés des fluides utilisées en dénominateur (comme la viscosité) sont $\ne 0$ pour éviter la division par zéro.

### 7.3. Validation de Plage de Température

Assure que la température d'entrée est positive et exprimée sur une échelle absolue (e.g., Kelvin), car $T > 0$ [17].

## 8. Références

[1] White, F. M. (2006). *Fluid Mechanics* (6th ed.). McGraw-Hill.

[2] Incropera, F. P., DeWitt, D. P., Bergman, T. L., & Lavine, A. S. (2007). *Fundamentals of Heat and Mass Transfer* (6th ed.). John Wiley & Sons.

[3] Cengel, Y. A., & Ghajar, A. J. (2015). *Heat and Mass Transfer: Fundamentals and Applications* (5th ed.). McGraw-Hill Education.

[4] Bejan, A. (2013). *Convection Heat Transfer* (4th ed.). John Wiley & Sons.

[5] Tritton, D. J. (1988). *Physical Fluid Dynamics* (2nd ed.). Oxford University Press.

[6] Bird, R. B., Stewart, W. E., & Lightfoot, E. N. (2007). *Transport Phenomena* (2nd ed.). John Wiley & Sons.

[7] Versteeg, H. K., & Malalasekera, W. (2007). *An Introduction to Computational Fluid Dynamics: The Finite Volume Method* (2nd ed.). Pearson Education.

[8] Schlichting, H., & Gersten, K. (2017). *Boundary-Layer Theory* (9th ed.). Springer.

[9] Blazek, J. (2015). *Computational Fluid Dynamics: Principles and Applications* (3rd ed.). Elsevier.

[10] Munson, B. R., Young, D. F., & Okiishi, T. H. (2009). *Fundamentals of Fluid Mechanics* (6th ed.). John Wiley & Sons.

[11] Kays, W. M., Crawford, M. E., & Weigand, B. (2005). *Convective Heat and Mass Transfer* (4th ed.). McGraw-Hill.

[12] Churchill, S. W., & Bernstein, M. (1977). A Correlation for Forced Convection from Gases and Liquids to a Circular Cylinder in Crossflow. *Journal of Heat Transfer*, 99(2), 300-306.

[13] Moran, M. J., Shapiro, H. N., Boettner, D. D., & Bailey, M. B. (2014). *Fundamentals of Engineering Thermodynamics* (8th ed.). John Wiley & Sons.

[14] Lemmon, E. W., Bell, I. H., & Huber, M. L. (2010). *NIST Standard Reference Database 23: Reference Fluid Thermodynamic and Transport Properties—REFPROP, Version 9.0* (Software). National Institute of Standards

[15] Fox, R. W., Pritchard, P. J., & McDonald, A. T. (2016). *Introduction to Fluid Mechanics* (9th ed.). John Wiley & Sons.

[16] Streeter, V. L., Wylie, E. B., & Bedford, K. W. (1998). *Fluid Mechanics* (9th ed.). McGraw-Hill
