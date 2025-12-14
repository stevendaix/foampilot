# Classe `Quantity` : Explication détaillée des méthodes

Ce document détaille chaque méthode de la classe `Quantity`, conçue pour gérer les grandeurs physiques avec unités.

## Introduction

La classe `Quantity` est une enveloppe autour de la bibliothèque Pint, permettant de manipuler des valeurs associées à des unités physiques. Elle facilite la conversion, l'affichage, la sérialisation, et les opérations arithmétiques.

## Constructeur

### Description

Le constructeur initialise une nouvelle instance de `Quantity` avec une valeur numérique et une unité.

- **Paramètres** :
  - `value` : Valeur numérique de la quantité.
  - `unit` : Chaîne de caractères représentant l'unité (ex: "m/s", "Pa", "kg").

## Méthodes principales

### `set_quantity`

#### Description

Met à jour la valeur et l'unité de la quantité.

- **Paramètres** :
  - `value` : Nouvelle valeur numérique.
  - `unit` : Nouvelle unité.

### `get_in`

#### Description

Convertit la quantité dans une unité cible et retourne sa valeur numérique.

- **Paramètres** :
  - `target_unit` : Unité cible pour la conversion.
- **Retourne** : Valeur numérique dans l'unité cible.
- **Exceptions** : Lève une erreur si la conversion est impossible.

### `to`

#### Description

Convertit la quantité dans une unité cible et retourne un nouvel objet `Quantity`.

- **Paramètres** :
  - `target_unit` : Unité cible pour la conversion.
- **Retourne** : Un nouvel objet `Quantity` avec la valeur convertie.
- **Exceptions** : Lève une erreur si la conversion est impossible.

## Sérialisation

### `as_dict`

#### Description

Retourne une représentation dictionnaire de la quantité.

- **Retourne** : Un dictionnaire avec les clés `value` et `unit`.
- **Exemple de sortie** : `{"value": 10.0, "unit": "meter / second"}`

### `from_dict`

#### Description

Crée un objet `Quantity` à partir d'un dictionnaire.

- **Paramètres** :
  - `data` : Dictionnaire contenant les clés `value` et `unit`.
- **Retourne** : Un nouvel objet `Quantity`.

### `from_pint`

#### Description

Crée un objet `Quantity` à partir d'une quantité Pint.

- **Paramètres** :
  - `pint_quantity` : Objet Pint Quantity.
- **Retourne** : Un nouvel objet `Quantity`.

## Représentation

### `__repr__` et `__str__`

#### Description

- **`__repr__`** : Retourne une représentation formelle de l'objet, utile pour le débogage.
- **`__str__`** : Retourne une représentation lisible de l'objet.

## Opérations arithmétiques

### Description

La classe supporte les opérations `+`, `-`, `*`, `/` avec d'autres objets `Quantity` ou des scalaires.

- **Addition (`__add__`)** : Additionne deux quantités ou une quantité et un scalaire.
- **Soustraction (`__sub__`)** : Soustrait deux quantités ou une quantité et un scalaire.
- **Multiplication (`__mul__`)** : Multiplie deux quantités ou une quantité et un scalaire.
- **Division (`__truediv__`)** : Divise deux quantités ou une quantité et un scalaire.

- **Retourne** : Un nouvel objet `Quantity` représentant le résultat de l'opération.

## Exemple d'utilisation

1. **Création d'objets** :
   - `speed = Quantity(10, "m/s")`
   - `pressure = Quantity(101325, "Pa")`

2. **Conversion** :
   - `speed.get_in("km/h")`
   - `pressure.get_in("atm")`

3. **Arithmétique** :
   - `d1 + d2`
   - `force / area`

4. **Sérialisation** :
   - `data = {"speed": speed.as_dict(), "pressure": pressure.as_dict()}`
   - `json_str = json.dumps(data, indent=2)`

5. **Désérialisation** :
   - `speed_loaded = Quantity.from_dict(loaded_data["speed"])`

## Conclusion

La classe `Quantity` simplifie la gestion des grandeurs physiques avec unités en Python, en s'appuyant sur Pint pour les conversions et en ajoutant des fonctionnalités de sérialisation et d'opérations arithmétiques.