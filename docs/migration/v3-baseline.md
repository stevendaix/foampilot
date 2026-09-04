# FoamPilot v3 — Baseline avant migration

## Objet

Cette baseline correspond à la branche `refactor/architecture-v3`, créée depuis `origin/main` au commit `dd060621bad2fd1c435065cb7a34dd394f0813fa` après la fusion de la PR #27. Cette première PR ne modifie pas l’architecture fonctionnelle : elle documente l’état de départ et prépare les outils d’audit réutilisables.

> Toute migration ultérieure devra comparer ses résultats à cette baseline afin de distinguer une régression introduite d’un problème préexistant.

## État des outils

| Élément | État lors de la baseline |
|---|---|
| Python | Disponible, Python 3.12 |
| pytest | Installé temporairement pour la baseline |
| Ruff | Non disponible dans l’environnement initial |
| Gmsh | Non disponible dans l’environnement initial |
| OpenFOAM Foundation 13 | Non disponible après la réinitialisation du sandbox |
| Dépôt propre | Oui, branche créée depuis `origin/main` |

## Baseline pytest

La commande exécutée est :

```bash
python3 -m pytest -q
```

Elle ne fournit pas encore un résultat fonctionnel exploitable : la collecte s’arrête avec `SystemExit: 2` dans `test_cfd_methods.py`, qui exécute `argparse.parse_args()` au chargement du module. Ce comportement empêche pytest de collecter l’ensemble de la suite lorsque des arguments pytest sont présents.

Résultat enregistré : `pytest_rc=3`, avec `77 errors during collection`. Cette observation est un **blocage de baseline**, pas une preuve que 77 tests fonctionnels échouent. La correction appartient à une future PR de stabilisation des tests : déplacer le parsing CLI dans une fonction `main()` ou protéger son exécution par `if __name__ == '__main__':`.

## Couverture à mesurer séparément

La baseline v3 doit ensuite être complétée par des commandes ciblées, sans les confondre avec le test global :

| Périmètre | Commande ou méthode | État |
|---|---|---|
| Tests unitaires Python | `pytest` par sous-répertoire | À exécuter après isolation des runners CLI |
| VMTK | tests et fixtures VMTK | À inventorier |
| Gmsh | tests de génération et conversion | Bloqué si Gmsh absent |
| OpenFOAM | runners et cas Foundation 13 | Bloqué si OpenFOAM absent |
| CHT | cas et tests `examples/cht` | À inventorier |
| Marine | tests et cas sous `openfoam13/` | À inventorier |
| Medical | tests `medical_build` et fixtures | À inventorier |
| FSI | PR #33 et #35 réservées | Hors périmètre de cette PR |
| Validations scientifiques | rapports et cas versionnés | À cartographier |

## Inventaire initial

L’outil `tools/audit/architecture_inventory.py` recense 3 110 fichiers dans le checkout de baseline. La répartition initiale est la suivante :

| Catégorie | Fichiers |
|---|---:|
| Exemples ou workflows historiques | 1 064 |
| Tutoriels | 780 |
| Documentation | 252 |
| Sources Python ou C++ | 228 |
| Dossiers versionnés OpenFOAM | 228 |
| Validation | 180 |
| Tests | 154 |
| Third-party | 105 |
| Autres | 119 |

Le compteur est un état de départ destiné à évoluer avec la migration. Les catégories se recouvrent partiellement par nature de chemin ; elles ne constituent pas encore une taxonomie définitive.

Le contrôle architectural initial détecte deux imports à traiter dans une future PR : `foampilot.urban.__init__` importe `foampilot.urban.validation`, et `foampilot.urban.geometry.gmsh_backend` importe `foampilot.urban.validation.geometry_checks`. Ces résultats ne sont pas corrigés dans la PR de baseline ; ils sont enregistrés comme dette architecturale préexistante.

## Règles de cette PR

Aucune suppression de code fonctionnel, aucun déplacement massif et aucune suppression de `openfoam13/` ne sont réalisés dans cette étape. La PR fournit uniquement la cartographie, les règles d’architecture et les outils d’audit nécessaires aux PR suivantes.
