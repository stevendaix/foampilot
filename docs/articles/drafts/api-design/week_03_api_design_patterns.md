# Pourquoi j'ai volontairement répété du code dans foampilot : la répétition intentionnelle comme vertu API

*En informatique, on nous apprend que "DRY" est sacré. Pourtant, dans foampilot, j'ai délibérément répété `write()` sur 8 classes. Voici pourquoi c'est un choix de design — et pourquoi ça rend l'API plus prévisible.*

---

## Introduction : L'hérésie du DRY

Dans le monde du développement logiciel, peu de principes sont aussi sacrés que **DRY** : *Don't Repeat Yourself*. "Si tu te répètes, factorise." C'est la première règle qu'on apprend, et elle est généralement juste.

Mais dans foampilot, j'ai fait le choix inverse. J'ai volontairement répété des motifs de code — notamment la méthode `write()` — sur plusieurs classes. Et c'est, je pense, l'une des meilleures décisions de design du projet.

Pourquoi ? Parce que dans le design d'APIs scientifiques, **la prévisibilité vaut mieux que la brièveté**.

---

## Le piège de l'abstraction excessive

Prenons un exemple concret. Au début du projet, j'aurais pu factoriser l'écriture des fichiers comme ça :

```python
# ❌ Ce que foampilot aurait pu faire
class CaseWriter:
    def write_solver_files(self, solver):
        # Écriture des fichiers solver
        pass
    
    def write_mesh_files(self, mesher):
        # Écriture des fichiers mesh
        pass
    
    def write_boundary_files(self, boundary):
        # Écriture des fichiers boundary
        pass
```

**Problème** : l'utilisateur doit apprendre 3 noms de méthodes différents pour faire la même chose fondamentale — écrire des fichiers. La documentation s'allonge, la surface de l'API grandit, et chaque nouvel utilisateur doit se demander : *"Est-ce que c'est `write_solver_files()` ou `write_case()` ?"*

C'est ce qu'on appelle l'**abstraction invisible** : elle semble élégante dans le code, mais elle rend l'API plus difficile à apprendre et à utiliser.

---

## Le `write()` universel : une interface prévisible

Dans foampilot, j'ai choisi la simplicité :

```python
# ✅ Ce que foampilot fait vraiment
class OpenFOAMFile:
    def write_file(self, filepath):
        # Écriture du dictionnaire

class SystemDirectory:
    def write(self):
        self.controlDict.write(system_path / 'controlDict')
        self.fvSchemes.write(system_path / 'fvSchemes')

class ConstantDirectory:
    def write(self):
        self.transportProperties.write(constant_path / 'transportProperties')
        self.turbulenceProperties.write(constant_path / 'turbulenceProperties')

class Meshing:
    def write(self):
        self.mesher.write()
```

Huit classes. Huit méthodes `write()`. Même signature. Même sémantique : *"persiste-toi toi-même sur le disque"*.

**L'avantage ?** Vous n'avez jamais à vous demander *"comment j'écris cet objet ?"*. La réponse est toujours la même :

```python
for component in [solver.system, solver.constant, meshing, boundary]:
    component.write()
```

C'est le **Command Pattern** dans sa forme la plus pure : chaque objet encapsule l'action de se persister lui-même. La répétition du nom `write()` n'est pas un défaut — c'est un **contrat d'interface** qui rend l'API apprenable en 10 minutes.

---

## `apply_condition_with_wildcard` : une signature pour les gouverner toutes

Même logique pour les conditions aux limites. Au lieu d'exposer 15 méthodes différentes (`set_velocity_inlet()`, `set_pressure_outlet()`, `set_wall()`, `set_symmetry()`, etc.), j'ai créé une méthode unique :

```python
# foampilot/boundaries/boundaries_dict.py
def apply_condition_with_wildcard(self, pattern: str, condition_type: str, **kwargs):
    for boundary in self.fields[next(iter(self.fields))].keys():
        if re.match(pattern, boundary):
            self.set_condition(boundary, condition_type, **kwargs)
```

Une seule signature pour toutes les conditions. L'utilisateur apprend une méthode, et il peut appliquer n'importe quelle condition :

```python
solver.boundary.apply_condition_with_wildcard("inlet", "velocityInlet", velocity=(10, 0, 0))
solver.boundary.apply_condition_with_wildcard("outlet", "pressureOutlet")
solver.boundary.apply_condition_with_wildcard("walls", "wall", friction=True)
```

**Réduction de la surface de l'API** : de 15 méthodes à 1. **Réduction de la charge cognitive** : l'utilisateur n'a plus à se souvenir du nom exact de chaque méthode.

---

## Pourquoi la répétition est une vertu ici

Dans la plupart des projets, la duplication de code est un anti-pattern. Dans foampilot, elle sert trois objectifs :

### 1. Découvrabilité

Quand chaque objet a un `write()`, vous n'avez pas à consulter la documentation pour trouver le bon nom de méthode. L'interface est **auto-descriptive**.

### 2. Composabilité

Des interfaces uniformes permettent des algorithmes génériques. Vous pouvez écrire :

```python
def write_all_case_files(objects):
    for obj in objects:
        obj.write()
```

Cette fonction fonctionne sur n'importe quelle combinaison d'objets foampilot, parce qu'ils partagent tous le même protocole. C'est la puissance du **polymorphisme par interface** — pas besoin de vérification de type (`isinstance`), pas besoin d'adapter chaque objet.

### 3. Réduction de la charge cognitive

Un ingénieur CFD a déjà beaucoup à retenir : schémas numériques, modèles de turbulence, maillage, conditions aux limites. Une API où tout se fait de la même façon élimine une variable mentale.

> **"La meilleure API est celle que vous n'avez pas besoin d'apprendre — parce que tout fonctionne comme prévu."**

---

## Quand factoriser, quand répéter ?

Ce choix de design n'est pas universel. Voici un tableau de décision :

| Situation | Factoriser | Répéter |
|-----------|-----------|---------|
| Logique métier identique | ✅ | ❌ |
| Interface utilisateur publique | ❌ | ✅ |
| Algorithme complexe | ✅ | ❌ |
| Point d'entrée public | ❌ | ✅ |
| Validation de données | ✅ | ❌ |
| Persistance (write/save) | ❌ | ✅ |

**Règle simple** : si ça concerne l'**utilisateur** (ce qu'il voit, ce qu'il appelle), préférez la répétition. Si ça concerne l'**implémentation** (ce que le code fait), factorisez.

---

## L'impact sur l'expérience développeur

J'ai mesuré l'impact de ce choix de plusieurs façons :

- **Temps d'apprentissage de l'API** : ~10 minutes pour un nouveau utilisateur
- **Nombre de méthodes `write()`** : 8 classes, 8 méthodes, même nom
- **Nombre de méthodes `apply_condition*`** : 1 méthode, 15 conditions supportées
- **Taux d'erreur "quelle méthode utiliser ?"** : quasi nul

Les retours des utilisateurs confirment : *"J'ai vu la documentation, j'ai vu le code, et j'ai compris immédiatement comment ça marche."*

C'est ça, la **DX (Developer Experience)** : rendre l'outil si prévisible qu'il disparaît. L'utilisateur ne pense plus à l'API, il pense à sa simulation.

---

## Conclusion : La prévisibilité comme vertu

Dans le design d'outils scientifiques, la simplicité n'est pas un luxe — c'est une nécessité. Un ingénieur CFD qui utilise votre API le fait pour résoudre un problème physique, pas pour apprendre une nouvelle interface.

En répétant intentionnellement `write()` et `apply_condition_with_wildcard`, j'ai créé une API où :
- **Tout s'écrit avec `.write()`**
- **Toute condition s'applique avec `.apply_condition_with_wildcard()`**

Pas de surprise. Pas de documentation à consulter pour chaque classe. Juste deux patterns à retenir, et puis c'est tout.

Dans le prochain article, je vous montre comment cette prévisibilité permet de visualiser OpenFOAM directement depuis Python — sans conversion, sans `foamToVTK`, sans douleur.

**Ressources :**
- Dépôt : [github.com/stevendaix/foampilot](https://github.com/stevendaix/foampilot)
- Documentation : [stevendaix.github.io/foampilot](https://stevendaix.github.io/foampilot/)

---

*Article en cours de amélioration — version 1.0*
