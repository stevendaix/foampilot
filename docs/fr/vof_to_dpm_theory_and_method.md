# Méthode VOF-to-DPM : théorie, algorithmes et implémentation OpenFOAM 13

**Statut :** documentation théorique et technique revue
**Qualification associée :** scénario multi-cloud/multi-espèces `NP=2` validé ; `NP=4` et les modèles réactifs restent hors qualification
**Version :** OpenFOAM 13, C++14
**Domaine :** transition de fragments liquides résolus par VOF vers des parcels lagrangiens DPM
**Auteur :** Manus AI

## 1. Objet et périmètre

La méthode VOF-to-DPM couple deux descriptions numériques complémentaires d’un écoulement multiphasique. La méthode **Volume of Fluid** décrit une interface liquide-gaz résolue sur une grille eulérienne. La méthode **Discrete Particle Model** décrit ensuite les fragments, gouttes ou particules sous forme de parcels lagrangiens transportés individuellement ou par groupes statistiques.

L’objectif de la transition n’est pas simplement de créer un parcel dans une cellule où `alpha` est non nul. Il faut déterminer une entité physique cohérente, éviter de la créer plusieurs fois en parallèle, transférer correctement sa masse, sa quantité de mouvement, son énergie et éventuellement sa composition, puis retirer exactement la quantité correspondante de la phase VOF. La transition est donc une **opération transactionnelle de changement de représentation**.

Cette documentation couvre la détection de fragments, la réconciliation des fragments traversant les frontières MPI, le choix du rang propriétaire, le Direct Commit, les confirmations et les critères de conservation. Elle ne constitue pas une qualification générale des modèles d’évaporation, de combustion, de collision ou de film liquide.

## 2. Descriptions physiques et numériques

### 2.1 Champ VOF

Dans VOF, chaque cellule contient une fraction volumique de phase, notée `alpha_l` pour la phase liquide. Pour une cellule de volume `V_c`, le volume liquide représenté est :

\[
V_{l,c} = \alpha_{l,c} V_c.
\]

Dans une formulation incompressible, la fraction volumique satisfait idéalement :

\[
0 \leq \alpha_l \leq 1.
\]

Pour deux phases, la fraction de la phase complémentaire est généralement :

\[
\alpha_g = 1-\alpha_l.
\]

La masse liquide de la cellule vaut :

\[
m_{l,c} = \rho_{l,c}\alpha_{l,c}V_c.
\]

Lorsque la densité liquide est considérée constante, on obtient :

\[
m_{l,c} = \rho_l\alpha_{l,c}V_c.
\]

La discrétisation VOF est adaptée aux interfaces continues et aux nappes liquides, mais elle devient coûteuse ou diffusive lorsque les gouttes deviennent beaucoup plus petites que la maille. Le passage vers DPM consiste à remplacer une composante liquide suffisamment détachée par un objet lagrangien.

### 2.2 Parcels lagrangiens

Un parcel DPM représente généralement une ou plusieurs particules physiques ayant des propriétés voisines. Sa position et sa vitesse suivent une description lagrangienne :

\[
\frac{d\mathbf{x}_p}{dt}=\mathbf{u}_p.
\]

Pour une particule sphérique soumise à la traînée, à la gravité et à d’autres forces :

\[
m_p\frac{d\mathbf{u}_p}{dt}
= \mathbf{F}_{drag} + \mathbf{F}_{gravity}
+ \mathbf{F}_{pressure}
+ \mathbf{F}_{added\ mass} + \cdots.
\]

Dans une approximation sphérique, le diamètre équivalent associé à un volume liquide `V_f` est :

\[
d_p = \left(\frac{6V_f}{\pi}\right)^{1/3}.
\]

La masse injectée dans le parcel est :

\[
m_p = \rho_l V_f.
\]

Si un parcel représente `N_p` particules physiques, la masse représentée est souvent écrite :

\[
m_{parcel}=N_p m_{particle}.
\]

Dans le prototype Direct Commit, le nombre de particules représentées est contrôlé explicitement par `nParticle`. La conversion ne doit pas modifier implicitement la masse en changeant le diamètre, la densité ou `nParticle` sans mettre à jour les bilans.

## 3. Équations de conservation du transporteur

### 3.1 Masse

Pour une phase liquide incompressible, la conservation continue de la masse est :

\[
\frac{\partial (\alpha_l\rho_l)}{\partial t}
+ \nabla\cdot(\alpha_l\rho_l\mathbf{u}_l)=0,
\]

hors termes de transfert de phase. Une conversion VOF-to-DPM introduit un terme de transfert `S_m` :

\[
\frac{\partial (\alpha_l\rho_l)}{\partial t}
+ \nabla\cdot(\alpha_l\rho_l\mathbf{u}_l)
= -S_m.
\]

Le cloud DPM reçoit le terme opposé :

\[
\frac{dM_{DPM}}{dt}=+\int_\Omega S_m\,d\Omega.
\]

Sur un pas `\Delta t`, la condition discrète de conservation est :

\[
\Delta M_{VOF} + \Delta M_{DPM}=0.
\]

Dans l’implémentation, la consommation du champ VOF n’est appliquée qu’après confirmation de la création du parcel. Un fragment rejeté ne doit donc jamais être soustrait du champ `alpha`.

### 3.2 Quantité de mouvement

La quantité de mouvement transportée par un fragment est approximée par :

\[
\mathbf{P}_f = m_f\mathbf{u}_f.
\]

Le parcel est initialisé avec une vitesse représentative du fragment, par exemple la moyenne pondérée par le volume liquide :

\[
\mathbf{u}_f
= \frac{\sum_{c\in f}\alpha_{l,c}V_c\mathbf{u}_c}
       {\sum_{c\in f}\alpha_{l,c}V_c}.
\]

Si la phase VOF est consommée au même pas temporel, la source de quantité de mouvement appliquée au transporteur doit être opposée au transfert du parcel. Le code actuel se concentre principalement sur la masse, l’énergie et la création atomique du parcel ; un cas de validation de quantité de mouvement doit être ajouté avant de revendiquer une conservation mécanique complète dans toutes les configurations.

### 3.3 Énergie et enthalpie

Pour une masse de fragment `m_f`, une température `T_f`, une pression `p_f` et une composition `Y_i`, l’enthalpie totale est :

\[
H_f = m_f h(T_f,p_f,\mathbf{Y}).
\]

Dans une approximation calorique simple :

\[
H_f \simeq m_f C_p(T_f-T_{ref}) + m_f h_{ref}.
\]

Cette approximation n’est acceptable que si elle est cohérente avec le modèle thermodynamique du cloud et du transporteur. Pour un mélange multi-composants, `C_p` et `h` peuvent dépendre de la composition :

\[
h(T,p,\mathbf{Y})=\sum_i Y_i h_i(T,p),
\qquad
C_p(T,p,\mathbf{Y})=\sum_i Y_i C_{p,i}(T,p).
\]

La conservation discrète de l’énergie lors de la transition est :

\[
\Delta H_{VOF} + \Delta H_{DPM}=0,
\]

en tenant compte de la convention de signe des sources. Une égalité de masse seule ne prouve donc pas la conservation énergétique.

### 3.4 Espèces

Pour `N_s` espèces, les fractions massiques satisfont :

\[
Y_i\geq0,
\qquad
\sum_{i=1}^{N_s}Y_i=1.
\]

La masse de l’espèce `i` dans un fragment vaut :

\[
m_{f,i}=m_fY_{f,i}.
\]

Le transfert correct doit conserver chaque composante :

\[
\Delta M_{VOF,i}+\Delta M_{DPM,i}=0,
\qquad i=1,\ldots,N_s.
\]

Dans le cas de validation actuel, les fractions sont configurées dans le fvModel puis propagées vers `directParcelData`. Cette stratégie valide le mécanisme transactionnel et la comptabilité par espèce, mais ne remplace pas encore une reconstruction de `Y_i` depuis des champs VOF indépendants ou un modèle réactif complet.

## 4. Détection d’un fragment VOF

### 4.1 Cellules candidates

Une cellule est candidate si elle satisfait, selon la configuration :

\[
\alpha_{min} \leq \alpha_{l,c} \leq \alpha_{max},
\]

et si son volume liquide dépasse un seuil numérique. Les seuils `alphaThreshold`, `minCells` et `minVolume` empêchent la création de parcels à partir de bruit numérique ou de résidus d’interface.

La sélection doit rester déterministe entre les rangs. Une cellule candidate n’est pas encore un fragment : elle doit être regroupée avec les candidates voisines appartenant à la même composante connexe.

### 4.2 Graphe de connexité

On définit un graphe :

\[
G=(V,E),
\]

où chaque sommet représente une cellule candidate et chaque arête relie deux cellules voisines. Une composante connexe du graphe est un fragment local.

Les connexions à travers une frontière `processor` sont traitées comme des arêtes distantes. Le rang local ne peut pas conclure qu’un fragment est terminé tant que les labels des cellules voisines du rang adjacent n’ont pas été échangés.

L’algorithme de type Union-Find effectue les opérations suivantes :

1. Chaque cellule candidate commence dans son propre ensemble.
2. Deux cellules voisines du même rang sont fusionnées.
3. Les labels des faces processor sont échangés avec les rangs voisins.
4. Les ensembles qui se touchent sur une face processor sont fusionnés.
5. Le représentant final devient l’identité de la composante globale.

La détection ne doit pas créer de parcel. Elle ne produit qu’un batch de décisions physiques candidates.

### 4.3 Propriétés du fragment

Pour chaque fragment `f`, on calcule au minimum :

\[
V_f=\sum_{c\in f}\alpha_{l,c}V_c,
\]

\[
m_f=\sum_{c\in f}\rho_{l,c}\alpha_{l,c}V_c,
\]

\[
\mathbf{x}_f
=\frac{1}{V_f}\sum_{c\in f}
\alpha_{l,c}V_c\mathbf{x}_c,
\]

\[
\mathbf{u}_f
=\frac{1}{m_f}\sum_{c\in f}
\rho_{l,c}\alpha_{l,c}V_c\mathbf{u}_c.
\]

Le fragment conserve deux listes de cellules :

| Liste | Usage |
|---|---|
| `globalCells` | identité, audit, reconstruction de la composante et diagnostic MPI |
| `localCells` | recherche de cellule, insertion du parcel et application des sources sur le rang local |

Mélanger ces deux listes est une erreur critique. `globalCells` peut contenir des indices appartenant à d’autres sous-domaines et ne doit jamais être utilisé directement comme index d’un champ local.

## 5. Identité déterministe et indépendance de la décomposition

Une identité de fragment basée sur le rang MPI ou sur l’ordre local de détection n’est pas stable. Elle change lorsque le maillage est décomposé différemment et peut provoquer des doublons ou des pertes.

L’implémentation utilise une numérotation globale des cellules indépendante de la décomposition, construite à partir d’un ordre déterministe des centres de cellules. L’identité du fragment est ensuite dérivée de cette représentation globale, par exemple à partir du plus petit identifiant global de cellule ou d’une combinaison ordonnée des cellules.

Les propriétés attendues sont :

\[
ID(f;D_1)=ID(f;D_2),
\]

pour une même géométrie physique et deux décompositions `D_1` et `D_2`, sous réserve de tolérances et de centres non ambigus.

Cette stabilité est nécessaire mais non suffisante. Il faut aussi que le choix du propriétaire, la composition du batch et la clé de confirmation soient déterministes.

## 6. Namespace logique par cloud et champ alpha

Dans un cas multi-cloud, `fragmentId` seul n’est pas une clé globale suffisante. Deux managers peuvent produire le même identifiant pour deux champs alpha différents. La clé complète est :

\[
K_f=(cloudName,alphaFieldName,fragmentId).
\]

Le namespace est généralement représenté par :

```text
namespaceKey = cloudName + "." + alphaFieldName
```

Exemples :

```text
waterCloud.alpha.water
fuelCloud.alpha.air
```

Cette clé est utilisée pour les objets auxiliaires enregistrés dans le `objectRegistry` :

```text
vofFragmentMask.waterCloud.alpha.water
vofFragmentMask.fuelCloud.alpha.air
vofLocalTransitionBatch.waterCloud.alpha.water
vofConfirmations.fuelCloud.alpha.air
```

Le nom physique du parcel cloud ne doit pas être modifié. `waterCloud` et `fuelCloud` doivent rester les noms recherchés par `parcelCloudList`. Le namespace concerne les champs, batches, stores et transactions du couplage VOF-to-DPM.

## 7. Protocole Direct Commit

### 7.1 Pourquoi contourner `InjectionModel`

Le chemin standard d’injection peut appeler des callbacks qui supposent un comportement symétrique de tous les rangs. Or, dans cette application, seul le rang propriétaire doit créer le parcel. Si un rang attend une collective qu’un autre rang ne peut pas atteindre dans le même ordre, le solveur peut rester bloqué.

Le Direct Commit sépare donc les opérations locales des opérations collectives :

| Étape | Nature | Collective MPI ? |
|---|---|---:|
| Détection locale | lecture de champs locaux | non, avant réconciliation |
| Fusion des fragments | échanges de frontières | oui |
| Construction du `directParcelData` | rang propriétaire | non |
| `addParticle()` | rang propriétaire | non |
| Écriture de confirmation | rang propriétaire | non |
| Réconciliation des confirmations | validation globale | oui |
| Consommation VOF | après confirmation | selon le modèle de sources |

### 7.2 Données de commit

Une structure de commit doit contenir les données physiques et transactionnelles :

```cpp
struct directParcelData
{
    word cloudName;
    label celli;
    point position;
    vector velocity;
    scalar diameter;
    scalar density;
    scalar temperature;
    scalar Cp;
    scalar nParticle;
    scalarList speciesMassFractions;
};
```

Le dispatch doit rechercher le cloud par nom et refuser explicitement un cloud inconnu. Un fallback implicite vers `clouds_[0]` est interdit dans une configuration multi-cloud.

### 7.3 Sélection du propriétaire

Le propriétaire est choisi de manière déterministe. Une règle possible consiste à choisir le rang contenant la cellule locale associée au plus petit identifiant global du fragment. Le propriétaire doit satisfaire :

\[
owner(f)\in\{0,\ldots,N_{proc}-1\}.
\]

Seul ce rang peut exécuter le commit. Les autres rangs conservent éventuellement une copie d’audit, mais n’appellent pas `addParticle()`.

### 7.4 Idempotence

Un commit doit être idempotent par clé `K_f`. Si la même transaction est présentée deux fois, le système doit la détecter avant de créer un second parcel. L’idempotence est essentielle en cas de répétition d’un callback, de reprise ou de divergence temporaire entre les étapes de publication et de confirmation.

## 8. Confirmations et transaction en deux phases

La transition suit un protocole en deux phases.

### Phase A : préparation et commit local

Le manager publie un batch contenant les fragments globaux et leurs propriétés. Le rang propriétaire construit le parcel, appelle `commitDirect()` et crée une confirmation locale :

```cpp
struct vofParcelConfirmation
{
    word cloudName;
    word alphaFieldName;
    uint64_t fragmentId;
    label ownerProc;
    label parcelsAdded;
    scalar massAdded;
    scalar expectedMass;
    scalarList speciesMassAdded;
    scalarList expectedSpeciesMass;
    bool success;
};
```

Une confirmation `success=false` ne doit jamais déclencher la consommation du champ VOF.

### Phase B : réconciliation globale

Les confirmations de tous les rangs sont rassemblées sur le rang maître, puis regroupées par :

```text
cloudName.alphaFieldName.fragmentId
```

Pour chaque fragment, la réconciliation vérifie :

\[
N_{confirmation}=1,
\]

\[
owner_{confirmation}=owner_{fragment},
\]

\[
M_{added}=M_{expected},
\]

\[
M_{added,i}=M_{expected,i}\quad\forall i.
\]

Le statut validé est ensuite diffusé aux rangs propriétaires. Cette diffusion concerne la décision de réconciliation ; elle ne signifie pas que tous les modèles thermo-réactifs OpenFOAM sont couverts. L’application des sources est réalisée uniquement pour les statuts valides.

## 9. Cycle d’un pas temporel

Le cycle recommandé dans le fvModel est :

```text
1. Vérifier que le timeIndex n’a pas déjà été traité.
2. Réinitialiser les champs de taux et les flags du pas.
3. Détecter et réconcilier les fragments VOF.
4. Publier le batch dans un objet namespacé.
5. Construire les parcels uniquement sur les rangs propriétaires.
6. Enregistrer les confirmations locales.
7. Réconcilier les confirmations avec MPI.
8. Appliquer les sources uniquement pour les confirmations valides.
9. Consommer alpha après confirmation.
10. Marquer le timeIndex comme traité.
```

Le batch doit être caché par `timeIndex` afin d’éviter qu’un même `correct()` ou qu’une seconde requête du solveur ne recrée les parcels. La réinitialisation doit vider le contenu du batch sans détruire et recréer continuellement l’objet enregistré dans le `objectRegistry`.

## 10. Conservation et audit

### 10.1 Bilan global

Sur un domaine `\Omega`, le bilan massique global doit vérifier :

\[
M_{prepared}=M_{created}=M_{confirmed}.
\]

Le bilan par cloud est :

\[
M_{prepared}^{(k)}=M_{created}^{(k)}=M_{confirmed}^{(k)}.
\]

Le bilan par espèce est :

\[
M_{prepared,i}^{(k)}=M_{created,i}^{(k)}=M_{confirmed,i}^{(k)}.
\]

### 10.2 Tolérances

Les tolérances doivent être relatives et absolues :

\[
|a-b|\leq \epsilon_{abs}+\epsilon_{rel}\max(|a|,|b|).
\]

Une tolérance d’audit ne doit pas compenser une mauvaise unité, un mauvais signe de source ou une double insertion. Elle doit seulement couvrir l’arrondi, la réduction MPI et la précision d’affichage.

### 10.3 Logs structurés

Chaque commit et confirmation doit être identifiable dans les logs :

```text
VOF direct commit cloud=waterCloud fragmentId=0 success=true mass=0.646099
VOF confirmation cloud=waterCloud alphaField=alpha.water fragmentId=0 success=true mass=0.646099 speciesMass=2(0.452269 0.193830)
```

Les auditeurs doivent détecter au minimum :

| Erreur | Signification |
|---|---|
| deux commits pour la même clé | duplication de parcel |
| confirmation sans commit | incohérence transactionnelle |
| commit sans confirmation | parcel non auditable |
| propriétaire divergent | routage MPI incorrect |
| cloud inconnu | dispatch dangereux |
| composition de taille différente | contrat thermo incompatible |
| somme des `Y_i` différente de 1 | composition invalide |
| source appliquée avant confirmation | perte potentielle de masse |

## 11. Implémentation OpenFOAM 13

### 11.1 `vofFragmentTransitionManager`

Le manager contient les champs de référence (`alpha`, `U`, `rho`), les seuils de détection, l’identité du cloud, l’identité du champ alpha, la clé de namespace, les fractions d’espèces et le dernier batch caché.

Son rôle est volontairement limité : détecter, agréger et réconcilier. Il ne doit pas appeler `addParticle()` et ne doit pas contenir de logique dépendante d’un rang propriétaire pendant la phase collective de détection.

### 11.2 `parcelCloud` et `parcelCloudList`

Le patch framework expose une méthode virtuelle `commitDirect()` dans la classe de base. L’implémentation template `ParcelCloud<CloudType>` construit le parcel avec la position, la cellule, le diamètre, la densité, la vitesse, la température et la composition, puis appelle `addParticle()`.

`parcelCloudList` effectue le dispatch par `cloudName`. Cette API est une extension locale nécessaire au portage ; elle doit être reconstruite avec `liblagrangianParcel` avant la compilation des bibliothèques VOF-to-DPM.

### 11.3 fvModel compressible et incompressible

Les deux modèles suivent le même protocole. Le modèle compressible ajoute en plus les bilans d’enthalpie et les sources thermodynamiques. Le modèle incompressible conserve le contrat de commit et de confirmation afin que la logique MPI ne diverge pas inutilement entre les deux variantes.

Les champs auxiliaires doivent être créés avec un nom dérivé de `namespaceKey`. Les champs physiques OpenFOAM tels que `U`, `p`, `T` et les champs alpha attendus par le solveur ne doivent pas être renommés par le couplage.

## 12. Cas multi-cloud et multi-composants

Un champ VOF unique ne suffit pas à distinguer deux liquides de composition différente. Il faut soit plusieurs champs alpha, soit un tag de phase explicite. Le cas de validation utilise deux champs alpha logiques et deux instances fvModel.

Pour deux clouds et deux espèces, une configuration typique est :

```text
waterCloud.alpha.water : Y = (0.70, 0.30)
fuelCloud.alpha.air    : Y = (0.20, 0.80)
```

Dans le cas livré, cette configuration teste :

1. le dispatch par cloud ;
2. l’absence de collision de registre ;
3. l’isolation des batches ;
4. l’isolation des confirmations ;
5. la conservation par espèce ;
6. l’absence de double insertion.

Elle ne qualifie pas automatiquement les modèles de réaction, d’évaporation, de changement de phase ou de diffusion des espèces. Ces modèles ajoutent des termes de source qui doivent être séparés du transfert initial VOF-to-DPM.

## 13. Validation recommandée

### 13.1 Tests unitaires

Les tests unitaires doivent vérifier les opérations suivantes sans solveur complet : construction d’une clé de namespace, égalité de deux clés, détection d’un doublon, rejet d’un cloud inconnu, rejet d’un vecteur d’espèces de taille incorrecte, conservation de la somme des fractions et idempotence d’un commit répété.

### 13.2 Tests MPI

La matrice minimale est :

| Test | Rangs | Objectif |
|---|---:|---|
| mono-cloud thermo | 1 | référence séquentielle |
| mono-cloud thermo | 2 | absence de deadlock Direct Commit |
| deux clouds, deux espèces | 2 | routage et confirmations composites |
| deux clouds, deux espèces | 4 | robustesse de décomposition |
| décomposition différente | 2 puis 4 | identité déterministe |
| propriétaire déplacé | 2 | cohérence du choix de rang |
| double commit injecté | 2 | idempotence et rejet du doublon |

### 13.3 Critères d’acceptation

Pour déclarer une configuration qualifiée, il faut vérifier au minimum :

- le solveur atteint `End` ;
- aucun rang ne reste bloqué dans une collective ;
- chaque transaction possède exactement un commit et une confirmation ;
- les masses préparées, créées et confirmées sont égales ;
- les masses de chaque espèce sont conservées ;
- aucun fallback vers le cloud par défaut n’est utilisé ;
- les résultats restent cohérents avec une autre décomposition ;
- les sources VOF ne sont appliquées qu’après confirmation.

## 14. Limites et extensions futures

La méthode dépend de la qualité de la détection VOF. Une interface trop diffuse, une composante plus petite qu’une cellule ou une topologie instable peut rendre le volume et le diamètre équivalent non physiques. Le seuil de détection ne doit donc pas être interprété comme une loi universelle de fragmentation.

La conversion vers un diamètre unique perd l’information sur la distribution interne des volumes, les déformations et les gradients thermiques du fragment. Pour un spray réaliste, une loi de distribution ou plusieurs parcels peuvent être nécessaires.

La conservation de l’énergie dépend fortement de l’initialisation thermodynamique du parcel. Les cas multi-composants avec `Y_i`, évaporation ou réaction exigent un modèle d’enthalpie cohérent de part et d’autre de l’interface de représentation.

Enfin, le Direct Commit contourne certains mécanismes automatiques du chemin standard d’injection. Chaque nouveau type de `CloudType`, `ParcelType` ou modèle thermo doit donc être compilé et validé séparément.

## 15. Références

[1]: https://doi.org/10.1016/0021-9991(81)90145-5 "Hirt et Nichols, Volume of Fluid (VOF) Method for the Dynamics of Free Boundaries"

[2]: https://doi.org/10.1016/0021-9991(92)90240-Y "Brackbill, Kothe et Zemach, A Continuum Method for Modeling Surface Tension"

[3]: https://www.openfoam.com/documentation/guides/latest/doc/guide-lagrangian.html "OpenFOAM Lagrangian particle tracking documentation"

[4]: https://openfoam.org/version/13/ "OpenFOAM Foundation version 13"

[5]: https://doi.org/10.1007/978-1-4899-1251-3 "Crowe et al., Multiphase Flows with Droplets and Particles"

[6]: https://doi.org/10.1016/j.compfluid.2014.10.012 "Review of Eulerian–Lagrangian methods for dispersed multiphase flow"
