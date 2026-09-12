# Audit exhaustif des équations JOS-3

## Périmètre

L’audit compare l’article de Takahashi et al. [1], le dépôt JOS-3 original [2] et la copie embarquée dans FoamPilot [3]. Il couvre la structure nodale, les bilans thermiques, les résistances de transfert, les capacités, les conductances, les signaux de thermorégulation, la production de chaleur et l’extension de surface CFD.

## Résultats principaux

| Élément | Résultat |
|---|---|
| Nombre d’états réellement résolus | 85 dans le code ; l’abstract de l’article indique 83 |
| Nombre de segments corporels | 17 |
| Températures cutanées physiologiques | 17, une par segment |
| Températures CFD locales | N, ajoutées par `DistributedSurfaceNetwork` |
| Schéma temporel | Backward difference implicite |
| Pertes sèches natives | Convection + rayonnement via température opérative |
| Pertes sèches externalisées | Mode FoamPilot `external_surface` |
| Validation copie/original | Réussie sur l’exemple officiel avec flux externe nul |

## Matrice équation–implémentation

| Équations article | Module/fonction | Statut |
|---|---|---|
| (1)–(8), bilans des 8 types de tissus | `jos3.py`, `matrix.py`, `construction.py` | Correspondance confirmée |
| (9)–(10), pertes sèches | `thermoregulation.py:dry_r`, `jos3.py:_run` | Correspondance confirmée en mode natif |
| (11)–(12), évaporation | `thermoregulation.py:wet_r`, `evaporation` | Correspondance confirmée |
| (13), respiration | `thermoregulation.py:resp_heatloss` | Correspondance confirmée |
| (14)–(16), solaire court | `jos3.py:_run` et paramètres solaires | À valider numériquement avec rayonnement non nul |
| (17)–(20), BSA | `construction.py` | Correspondance confirmée |
| (21), capacités | `construction.py:capacity` | Correspondance confirmée, conversion Wh/K → J/K vérifiée |
| (22)–(23), conduction | `construction.py:conductance` | Correspondance confirmée |
| (24)–(28), signaux | `thermoregulation.py:error_signals` | Correspondance confirmée |
| (29)–(32), production de chaleur | `basal_met`, `local_mbase`, `local_mwork` | Correspondance confirmée |
| (33)–(35), frisson | `thermoregulation.py:shivering` | Correspondance confirmée, options à tester |
| (36)–(39), NST/BAT | `thermoregulation.py:nonshivering` | Correspondance globale ; branche d’âge suspecte |
| (44), (45), (50), vieillissement | `thermoregulation.py`, `jos3.py` | À valider par scénarios d’âge |

## Écarts identifiés

### 1. 83 contre 85 états

L’article contient une incohérence entre l’abstract et la section de construction. Le code original définit `NUM_NODES = 85` et les matrices ont une dimension 85. La copie FoamPilot conserve cette dimension. Aucun changement ne doit être fait vers 83 sans modifier toute la topologie et les matrices.

### 2. AVA

Le dépôt officiel documente une correction historique des équations de débit AVA main/pied. Le code actuel contient la formulation corrigée. Les versions anciennes ou une réécriture indépendante doivent être comparées à cette version, pas seulement à l’équation imprimée dans une ancienne publication.

### 3. `Mnshiv`

Le dépôt explique que le terme `Mnshiv` mentionné dans l’article n’est pas utilisé directement pour le débit sanguin ; le code utilise `Mwork + Mshiv`. Cette différence est intentionnelle et documentée par les auteurs.

### 4. Branche d’âge NST

Dans `nonshivering`, deux branches successives testent `age < 50`. La seconde ne peut jamais être atteinte. Ce point est un défaut de code potentiel, mais il ne faut pas le corriger sans identifier la tranche d’âge voulue dans la source scientifique ou les versions officielles ultérieures.

### 5. Extension de surface distribuée

La température indépendante par face CFD n’existe pas dans JOS-3 original. FoamPilot ajoute un réseau `N` états, avec capacités et conductances réparties par aire. Cette extension est conservatrice par zone mais n’est pas une équation publiée de JOS-3. Elle doit être validée séparément.

### 6. Mode `external_surface`

FoamPilot désactive la perte sèche interne `C+R` de JOS-3 et la délègue au réseau de surface CFD. Cette opération est nécessaire pour éviter un double comptage, mais elle signifie que le résultat n’est plus strictement celui du JOS-3 original lorsque le flux CFD est non nul.

## Validation disponible

La reproduction de l’exemple officiel compare les séries du modèle original, de la copie FoamPilot et du couplage à flux externe nul. Les écarts observés sur `TskMean`, `TskHead`, `TskChest`, `TskLFoot` et `TcrChest` sont nuls dans cette comparaison.

Cette validation démontre l’équivalence logicielle du chemin natif, mais pas encore la validation expérimentale de l’article. Pour reproduire cette dernière, il faut implémenter les cas des tableaux expérimentaux de l’article et disposer des mesures de référence.

## Références

[1]: https://doi.org/10.1016/j.enbuild.2020.110575 "Takahashi et al. 2021, Thermoregulation model JOS-3 with new open source code"
[2]: https://github.com/TanabeLab/JOS-3 "JOS-3 source code and documentation"
[3]: https://github.com/stevendaix/foampilot "FoamPilot source code"
