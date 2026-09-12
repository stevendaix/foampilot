# Brief — Semaine 3 : "La répétition intentionnelle dans les APIs scientifiques"

**Objectif** : Expliquer pourquoi la répétition de `write()` et de signatures constantes est un choix de design délibéré.
**Angle** : Deep-dive technique, design d'API, philosophie logicielle.

---

## Structure proposée

### Introduction — L'anti-pattern que tout le monde évite

Storytelling : On m'a toujours appris que "DRY" (Don't Repeat Yourself) est sacré. Pourtant, dans foampilot, j'ai volontairement répété `write()` sur 8 classes différentes. Pourquoi ?

**Promesse** : Comprendre quand la factorisation nuit à l'utilisateur, et quand la répétition sert l'API.

### Section 1 — Le piège de l'abstraction excessive

Montrer un exemple "mal factorisé" :

```python
# ❌ Ce que foampilot aurait pu faire
class CaseWriter:
    def write_solver_files(self, solver): ...
    def write_mesh_files(self, mesher): ...
    def write_boundary_files(self, boundary): ...
```

**Problème** : L'utilisateur doit apprendre 3 noms de méthodes différents pour faire la même chose (écrire).

### Section 2 — Le `write()` universel

Présenter le pattern réel :

```python
# ✅ Ce que foampilot fait vraiment
class Solver:
    def write_case(self):
        self.system.write()
        self.constant.write()

class SystemDirectory:
    def write(self):
        self.controlDict.write(system_path / 'controlDict')
        self.fvSchemes.write(system_path / 'fvSchemes')

class OpenFOAMFile:
    def write_file(self, filepath):
        # écriture du dictionnaire
```

**Avantage** : `obj.write()` marche toujours. Pas de documentation à consulter.

### Section 3 — Le pattern Command

Expliquer le Command Pattern formel :

- Chaque objet encapsule l'action de se persister
- Interface uniforme = composabilité
- Exemple : itérer sur une liste d'objets hétérogènes

```python
components = [solver.system, solver.constant, meshing, boundary]
for c in components:
    c.write()  # marche sur tous, sans instanceof
```

### Section 4 — `apply_condition_with_wildcard` : signature constante

Montrer comment une seule signature remplace 15 méthodes différentes :

```python
# ❌ Sans wildcard
boundary.set_velocity_inlet(...)
boundary.set_pressure_outlet(...)
boundary.set_wall(...)
boundary.set_symmetry(...)

# ✅ Avec wildcard
boundary.apply_condition_with_wildcard("inlet", "velocityInlet", velocity=(10,0,0))
boundary.apply_condition_with_wildcard("outlet", "pressureOutlet")
boundary.apply_condition_with_wildcard("walls", "wall", friction=True)
```

**Leçon** : Une signature constante réduit la surface de l'API et accélère l'apprentissage.

### Section 5 — Quand factoriser, quand répéter ?

Tableau de décision :

| Situation | Factoriser | Répéter |
|-----------|-----------|---------|
| Logique métier identique | ✅ | ❌ |
| Interface utilisateur | ❌ | ✅ |
| Algorithme complexe | ✅ | ❌ |
| Point d'entrée public | ❌ | ✅ |
| Validation | ✅ | ❌ |

### Section 6 — L'impact sur l'expérience développeur

Chiffres et témoignages :
- Temps d'apprentissage de l'API : ~10 minutes
- Nombre de méthodes `write()` : 8
- Nombre de méthodes `apply_condition*` : 1
- Réduction du taux d'erreur : significative (pas de "quelle méthode utiliser ?")

### Conclusion — La prévisibilité comme vertu

CTA : *"Maintenant que vous comprenez pourquoi l'API est prévisible, je vous montre comment elle permet de visualiser OpenFOAM sans outils externes."*

---

## Code à préparer

- [ ] Extrait du `write()` universel (8 classes)
- [ ] Comparaison avant/après (mal factorisé vs bien factorisé)
- [ ] Extrait de `apply_condition_with_wildcard`
- [ ] Tableau de décision factoriser vs répéter

## Images à préparer

- [ ] Schéma UML simplifié montrant les 8 classes avec `write()`
- [ ] Diagramme de flux : "utilisateur écrit `obj.write()` → fichier généré"
- [ ] Capture d'écran de la doc API (avant/après)

## Sources à citer

- Design Patterns: Elements of Reusable Object-Oriented Software (GoF)
- "The Art of Readable Code" (Dustin Boswell)
- Philosophie Python : "Explicit is better than implicit"
