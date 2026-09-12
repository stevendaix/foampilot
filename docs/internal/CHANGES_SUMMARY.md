# Résumé des Modifications foampilot — Aéraulique Urbaine

Date : 2026-08-05

---

## 1. Bugs Corrigés

### 1.1 Écriture multi-ligne de `codedFixedValue` — `openFOAMFile.py`

**Fichier** : `foampilot/src/foampilot/base/openFOAMFile.py`

**Problème** : Les blocs de code C++ multi-lignes dans les conditions aux limites `codedFixedValue` (champ `code`) n'étaient pas écrits correctement. La méthode `_write_attributes` et `write_boundary_file` ajoutaient un `;` après la dernière ligne du bloc `#{...#}`, ce qui cassait le format OpenFOAM.

**Correction** : Les méthodes `_write_attributes` (ligne 162) et `write_boundary_file` (ligne 278) détectent maintenant les valeurs multi-lignes (contenant `\n`) et les écrivent sans ajouter de `;` supplémentaire, car le point-virgule doit être à l'intérieur du délimiteur `#};`.

**Exemple de sortie correcte** :
```openfoam
code            #{
    const vector& pos = pos();
    scalar z = pos.z;
    scalar u_star = ...;
    result = vector(u_mag, 0, 0);
#};
```

---

### 1.2 Import incorrect dans `post_rotation.py`

**Fichier** : `examples/building_aero/old/post_rotation.py`

**Problème** : Le fichier importait `from lawson import (...)` mais le module réel est `foampilot.postprocess.wind_analysis`.

**Correction** :
```python
# Avant
from lawson import (
    WindRose, WindCaseResult, WindEnsemble,
    LawsonProcessor, LawsonVisualizer,
)

# Après
from foampilot.postprocess.wind_analysis import (
    WindRose, WindCaseResult, WindEnsemble,
    LawsonProcessor, LawsonVisualizer,
)
```

---

### 1.3 Bug regex dans `generate_wind_cases.py`

**Fichier** : `examples/building_aero/generate_wind_cases.py` (ligne 250)

**Problème** : Le pattern regex `rf'({patch_name}\s*\{{\s*type\s+)patch(;'` contenait une accolade `{` non échappée, provoquant une erreur `re.error: missing ), unterminated subpattern`.

**Correction** :
```python
# Avant
pattern = rf'({patch_name}\s*\{{\s*type\s+)patch(;'

# Après
pattern = rf'({patch_name}\s*\{{\s*type\s+)patch(;)'
```

---

### 1.4 Conversion `ValueWithUnit` → `float` pour `nu`

**Fichier** : `examples/building_aero/generate_wind_cases.py` (ligne 375)

**Problème** : `nu` est un objet `ValueWithUnit`, pas un `float`. L'appel `float(nu)` échoue avec `TypeError`.

**Correction** :
```python
# Avant
"nu": float(nu),

# Après
"nu": float(nu.get_in("m^2/s") if hasattr(nu, 'get_in') else nu),
```

---

### 1.5 Syntaxe `codedFixedValue` — `#};` au lieu de `#}`

**Fichier** : `examples/building_aero/generate_wind_cases.py` (lignes 312, 333, 351, 370)

**Problème** : Les blocs de code `codedFixedValue` se terminaient par `#}` mais OpenFOAM attend `#};` (le point-virgule est à l'intérieur du délimiteur de fermeture).

**Correction** : Tous les blocs de code ont été mis à jour :
```python
# Avant
"code": "#{\n" + u_code + "\n#}",

# Après
"code": "#{\n" + u_code + "\n#};",
```

---

## 2. Améliorations de Performance

### 2.1 Augmentation de `lc_max` pour maillage plus rapide

**Fichier** : `examples/building_aero/generate_wind_cases.py` (ligne 64)

**Modification** :
```python
# Avant
"maillage": {
    "lc_min": 5.0,
    "lc_max": 10.0,
},

# Après
"maillage": {
    "lc_min": 5.0,
    "lc_max": 50.0,
},
```

**Impact** : Réduction significative du nombre d'éléments dans les régions loin des bâtiments (far-field), accélérant à la fois la génération du maillage Gmsh et la simulation OpenFOAM. La résolution fine (`lc_min=5.0`) est conservée autour des bâtiments.

---

## 3. Conditions Auxiliaires Manquantes Ajoutées

**Fichier** : `examples/building_aero/generate_wind_cases.py`

Les conditions aux limites suivantes pour le champ `p` (pression) à l'entrée étaient manquantes, provoquant des erreurs `keyword type is undefined` lors de `decomposePar` :

```python
# Ajouté après la configuration de l'outlet
solver.boundary.set_raw_condition("INLET", "p", {"type": "zeroGradient"})
```

De même, les champs turbulence `omega` et `nut` à l'entrée n'avaient pas de conditions définies :

```python
# omega inlet — codedFixedValue avec profil logarithmique
solver.boundary.set_raw_condition("INLET", "omega", {
    "type": "codedFixedValue",
    "value": "uniform 0",
    "code": "#{\n" + omega_code + "\n#};",
})

# nut inlet — zeroGradient
solver.boundary.set_raw_condition("INLET", "nut", {"type": "zeroGradient"})
```

---

## 4. Vérifications Effectuées

| Vérification | Statut | Détails |
|---|---|---|
| Génération du maillage Gmsh | ✅ OK | 300 540 nœuds, 1 739 867 tétraèdres |
| `checkMesh` | ✅ OK | « Mesh OK. » — pas de volumes négatifs, aspect ratio < 10 |
| `decomposePar` (4 proc) | ✅ OK | 4 sous-domaines créés avec succès |
| `simpleFoam` (série) | ⚠️ Partiel | Démarre mais échoue sur `codedFixedValue` (lib manquante) |
| `simpleFoam` (parallèle) | ⚠️ Partiel | Même problème que série |
| Syntaxe `codedFixedValue` | ✅ OK | Format `#{...#};` correctement écrit |

---

## 5. Problème Connu : `codedFixedValue` Library

L'installation OpenFOAM 13 utilisée ne dispose pas de la bibliothèque `libcodedFixedValue.so` compilée. Cette bibliothèque est nécessaire pour utiliser le type de condition aux limites `codedFixedValue`.

**Solutions possibles** :
1. Compiler la bibliothèque depuis les sources :
   ```bash
   cd $WM_PROJECT_DIR/src/functionObjects/utilities/codedFunctionObject
   wmake
   ```
2. Utiliser `fixedValue` avec des valeurs uniformes pour les tests
3. Utiliser une version d'OpenFOAM qui inclut cette bibliothèque par défaut

---

## 6. Fichiers Modifiés

| Fichier | Modification |
|---|---|
| `foampilot/src/foampilot/base/openFOAMFile.py` | Support multi-ligne pour `codedFixedValue` |
| `examples/building_aero/old/post_rotation.py` | Correction import `lawson` → `wind_analysis` |
| `examples/building_aero/generate_wind_cases.py` | Regex fix, `#};` syntax, inlet BCs, `ValueWithUnit` fix, `lc_max=50` |
| `examples/building_aero/buildings_config.json` | `lc_max` mis à jour à 50.0 |

---

## 7. Prochaines Étapes

1. Compiler `libcodedFixedValue.so` pour OpenFOAM 13
2. Relancer la simulation parallèle avec `mpirun -np 4 simpleFoam -parallel`
3. Exécuter `reconstructPar` pour consolider les résultats
4. Lancer le post-traitement avec `wind_postprocess.py`
5. Générer les cartes Lawson et les visualisations
