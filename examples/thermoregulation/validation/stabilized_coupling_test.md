# Test du couplage JOS-3 stabilisé

## Modifications appliquées

Le générateur `prepare_fields.py` impose maintenant `fixedFluxPressure` sur `p_rgh` pour la face `human`. Le pas CFD est réduit à `deltaT=0.01 s`, avec `maxCo=0.5` et `maxDeltaT=0.01`. Le pilote JOS-3 applique une sous-relaxation de température `alpha=0.1` avant d’écrire `data.in`. Les réglages PIMPLE ont été conservés dans la configuration stable : prédicteur de quantité de mouvement désactivé, un correcteur non orthogonal nul, relaxation `p_rgh=0.3`, `U=0.1`, `h=0.1`.

Une tentative plus agressive avec `momentumPredictor yes`, deux correcteurs PIMPLE, un correcteur non orthogonal et `relTol=0` a divergé dès le premier pas. Elle n’est donc pas retenue.

## Test exécuté

Le cas humain de référence, avec cavité fermée et température externe fournie par JOS-3, a été exécuté de `t=0` à `t=1 s`. Le maillage contient 89 604 cellules et 20 223 faces humaines.

| Indicateur | Résultat |
|---|---:|
| Statut OpenFOAM | Réussi, `End` |
| Statut pilote JOS-3 | Réussi, `FoamPilot JOS-3 driver terminé.` |
| Faces échangées | 20 223 par pas |
| Coefficient `h` | 2,028 à 51,36 W/m²/K |
| Température cible JOS-3 finale | 30,00 à 34,15 °C |
| Température retournée sous-relaxée finale | 30,19 à 34,14 °C |
| Erreur de continuité globale finale | environ `6e-18` cumulée |
| Temps CPU | environ 110 s pour 1 s CFD |

Le résultat montre que la combinaison `fixedFluxPressure` + cavité fermée + pas de temps réduit + sous-relaxation `alpha=0.1` stabilise le couplage sur le maillage humain.

## Limite actuelle

Le test réussi utilise une cavité fermée. Il ne valide pas encore la configuration initiale avec inlet/outlet ouverts, qui diverge même à température fixe. La prochaine étape doit donc remplacer ces frontières par des conditions cohérentes d’enceinte ventilée, par exemple une condition pression/vitesse de type entrée-sortie avec `prghTotalPressure` et `pressureInletOutletVelocity`, ou conserver la cavité fermée comme cas de validation thermorégulation de base.
