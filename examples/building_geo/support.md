Oui. Et pour ton besoin, **un DWG n’est probablement pas la meilleure donnée de départ**. En France, il existe aujourd’hui des données publiques qui permettent de reconstruire un quartier en 3D de manière assez précise, notamment pour un usage CFD / aérodynamique urbaine.

### Ce que tu peux obtenir

Pour chaque bâtiment, l'objectif serait d'avoir quelque chose comme :

| Donnée              |                             Exemple |
| ------------------- | ----------------------------------: |
| Emprise au sol      |                         polygone XY |
| Longueur            |                                42 m |
| Largeur             |                                18 m |
| Hauteur             |                              12.4 m |
| Altitude du terrain |                              86.2 m |
| Nombre d'étages     |                      éventuellement |
| Forme du bâtiment   |                 polygone quelconque |
| Orientation         |                                 27° |
| Toiture             | éventuellement pente / forme réelle |

La meilleure combinaison que je vois est :

**1. Cadastre → empreintes précises des bâtiments**
**2. IGN BD TOPO → géométrie + hauteur des bâtiments**
**3. IGN LiDAR HD → hauteur et forme réellement mesurées**
**4. Orthophoto → contrôle visuel**
**5. Génération automatique → STEP/STL/DXF/OBJ/OpenFOAM**

Le cadastre fournit notamment les polygones des bâtiments et des parcelles, en données ouvertes. ([Données cadastrales ouvertes][1])

Mais surtout, **la BD TOPO est très intéressante pour ton cas** : l'IGN indique explicitement que sa description est 3D et qu'elle fournit **l'altimétrie des objets ainsi que la hauteur des bâtiments**. ([data.gouv.fr][2])

---

## 1. BD TOPO : probablement le meilleur point de départ

Tu peux récupérer la couche **Bâtiment** de la BD TOPO.

Elle contient des géométries vectorielles représentant les bâtiments, avec notamment les informations permettant de connaître leur hauteur.

[BD TOPO sur data.gouv.fr](https://www.data.gouv.fr/datasets/bd-topo-r?utm_source=chatgpt.com)

L'intérêt est que tu n'as **pas besoin d'avoir un DWG fourni par la mairie**.

Tu peux partir directement de :

```text
              Bâtiment A
          ┌───────────────┐
          │               │
          │               │  H = 12.5 m
          │               │
          └───────────────┘
               ↑
          polygone XY
```

et extruder automatiquement le polygone suivant sa hauteur.

Par exemple :

```python
building = {
    "geometry": [
        (0, 0),
        (42, 0),
        (42, 18),
        (0, 18)
    ],
    "height": 12.5
}
```

pour générer :

```text
z = 12.5 m
┌──────────────────────┐
│                      │
│                      │
└──────────────────────┘
z = 0 m
```

---

# 2. Mais pour ton objectif, je regarderais surtout le LiDAR HD

C'est potentiellement **beaucoup plus intéressant que la hauteur moyenne de la BD TOPO**.

Le LiDAR permet de mesurer directement la surface du quartier :

```text
                   toiture
              ● ● ● ● ● ● ●
           ● ● ● ● ● ● ● ● ●
         ● ● ● ● ● ● ● ● ● ●
         │                  │
         │     bâtiment     │
         │                  │
         │                  │
─────────┴──────────────────┴──────── sol
```

Tu peux alors obtenir :

* hauteur des bâtiments ;
* variation de hauteur ;
* forme des toitures ;
* relief ;
* arbres ;
* mobilier urbain éventuellement ;
* bâtiments voisins ;
* géométrie beaucoup plus réaliste.

L'IGN dispose également du RGE, qui regroupe notamment **BD TOPO, BD ORTHO et les données altimétriques**. ([data.gouv.fr][3])

La BD ORTHO est particulièrement utile pour vérifier les emprises et la géométrie, avec une résolution de l'ordre de 20 cm. ([data.gouv.fr][4])

---

# 3. Il existe même des données déjà en volumes 3D

C'est intéressant pour ton problème.

Par exemple, la Ville de Paris diffuse un jeu de données appelé **« Volumes bâtis »**.

Il décrit les bâtiments en différenciant les différentes hauteurs et les parties en saillie/retrait pour construire directement des **volumes 3D**. ([data.gouv.fr][5])

Donc selon le quartier que tu veux modéliser, il peut exister des données locales **encore meilleures que la BD TOPO nationale**.

---

# 4. Et le DWG ?

Le DWG n'est pas forcément disponible pour tous les quartiers.

Le plus souvent tu vas rencontrer :

```text
SHP
GeoJSON
GPKG
DXF
LAS / LAZ
```

plutôt que :

```text
DWG
```

Mais ce n'est absolument pas bloquant.

Par exemple :

```text
BD TOPO / Cadastre
        ↓
     GeoJSON
        ↓
      Python
        ↓
   polygones XY
        +
     hauteur
        ↓
   extrusion 3D
        ↓
      STL
        ↓
   OpenFOAM
```

ou :

```text
GeoJSON
   ↓
QGIS
   ↓
DXF
```

si tu veux travailler ensuite dans SpaceClaim/ANSYS.

---

# 5. Pour ton usage CFD, je ferais quelque chose de plus ambitieux

Vu que ton objectif semble être de **reconstruire un quartier pour une simulation aérodynamique**, je ne chercherais pas simplement un fichier DWG.

Je construirais un pipeline :

```text
             IGN / Cadastre
                   │
       ┌───────────┼────────────┐
       ↓           ↓            ↓
   BD TOPO      Cadastre     LiDAR HD
       │           │            │
       └───────────┼────────────┘
                   ↓
          reconstruction 3D
                   ↓
          ┌─────────────────┐
          │   buildings     │
          │   roads         │
          │   terrain       │
          │   vegetation    │
          └─────────────────┘
                   ↓
              nettoyage
                   ↓
           simplification
                   ↓
            maillage CFD
                   ↓
              OpenFOAM
```

Et surtout, **je conserverais les bâtiments sous forme de polygones + attributs**, plutôt que de transformer immédiatement tout en STL.

Cela permettrait ensuite de faire automatiquement :

```text
building_id
x
y
height
orientation
area
perimeter
roof_type
...
```

puis de générer plusieurs niveaux de simplification.

---

## 6. On peut même avoir une représentation très simple pour CFD

Par exemple, pour un quartier :

```text
              vent
               ↓
               ↓
       ┌───────┐       ┌───────────┐
       │       │       │           │
       │ B001  │       │   B002    │
       │ 15 m  │       │    23 m   │
       └───────┘       └───────────┘

 ┌───────────────┐
 │               │
 │     B003      │
 │     8 m       │
 │               │
 └───────────────┘

             ┌────────────┐
             │    B004    │
             │    31 m    │
             └────────────┘
```

Et ensuite générer automatiquement un domaine :

```text
               inlet
                 ↓
 ┌──────────────────────────────────────┐
 │                                      │
 │       ████            ███████        │
 │       ████      ███   ███████        │
 │       ████      ███   ███████        │
 │                 ███                  │
 │       ███████████████                │
 │                                      │
 │                                      │
 └──────────────────────────────────────┘
```

avec les hauteurs exactes.

---

# 7. Ce que je te propose

Si tu me donnes **le quartier précis** — par exemple :

> « le quartier autour de telle adresse à Lyon »

ou simplement :

> **une adresse / coordonnées GPS / commune + rues**

je peux chercher **les données réellement disponibles pour cette zone** et déterminer :

1. **empreintes des bâtiments disponibles**
2. **hauteur de chaque bâtiment**
3. **possibilité d'obtenir les toitures**
4. **présence de LiDAR HD**
5. **format disponible : SHP / GeoJSON / GPKG / DXF / LAS / LAZ**
6. **précision attendue**
7. **méthode pour transformer tout ça en géométrie 3D**
8. et surtout **comment l'exporter vers OpenFOAM / Gmsh / SpaceClaim**.

On pourrait même faire un petit script Python qui prend simplement :

```text
adresse du quartier
+
rayon = 500 m
```

et produit automatiquement :

```text
quarter/
├── buildings.geojson
├── buildings.csv
├── buildings.step
├── buildings.stl
├── buildings.dxf
└── openfoam/
    └── ...
```

**Si tu me donnes le quartier que tu veux reconstruire, je peux chercher directement les données disponibles et te dire exactement jusqu'à quel niveau de détail on peut aller.**

[1]: https://cadastre.data.gouv.fr/?utm_source=chatgpt.com "Données cadastrales ouvertes | cadastre.data.gouv.fr"
[2]: https://www.data.gouv.fr/datasets/bd-topo-r?utm_source=chatgpt.com "Jeu de données - BD TOPO® | data.gouv.fr"
[3]: https://www.data.gouv.fr/datasets/referentiel-a-grande-echelle-rge?utm_source=chatgpt.com "Jeu de données - Référentiel à grande échelle (RGE) | data.gouv.fr"
[4]: https://www.data.gouv.fr/datasets/bd-ortho-r?utm_source=chatgpt.com "Jeu de données - BD ORTHO® | data.gouv.fr"
[5]: https://www.data.gouv.fr/datasets/volumes-batis?utm_source=chatgpt.com "Jeu de données - Volumes bâtis | data.gouv.fr"
