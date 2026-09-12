 build123dLogo

dernier
 
Rechercher des documents
Introduction
Installation
Concepts clés
Concepts clés (mode constructeur)
Concepts clés (mode algèbre)
Objets en mouvement
Transition depuis OpenSCAD
Exemples introductifs
1. Plaque rectangulaire simple
2. Plaque avec trou
3. Un solide prismatique extrudé
4. Construction de profils à l'aide de lignes et d'arcs
5. Déplacement du point de travail actuel
6. Utilisation des listes de points
7. Polygones
8. Polylignes
9. Sélecteurs, filets et chanfreins
10. Sélectionnez Dernier et Trou
11. Utilisez une face comme plan pour BuildSketch et introduisez GridLocations.
12. Définition d'un bord avec une spline
13. Contre-trous, contre-trous et emplacements polaires
14. Positionnez-vous sur une ligne avec '@', '%' et introduisez le balayage
15. Géométrie symétrique par réflexion
16. Mise en miroir d'objets 3D
17. Effet miroir sur les visages
18. Création de placements sur les visages
19. Localisation d'un placement sur un sommet
20. Placement décalé de l'esquisse
21. Créez un plan au centre d'une autre forme.
22. Placements rotatifs
23. Tourner
24. Loft
25. Croquis décalé
26. Décaler une pièce pour créer des éléments fins
27. Division d'un objet
28. Localisation des caractéristiques à partir des visages
29. La bouteille classique OCC
30. Courbe de Bézier
31. Lieux de nidification
32. Boucle for en Python
33. Fonction Python et boucle for
34. Texte en relief et en creux
35. Machines à sous
36. Extruder jusqu'à
37. Positionnement des croquis dans un plan
Tutoriels
Objets
Opérations
Sélection et exploration de la topologie
constructeurs
Articulations
Assemblées
Conseils, bonnes pratiques et FAQ
Import/Export
Sujets avancés
Aide-mémoire
Outils et bibliothèques externes
Référence de l'API commune du générateur
Référence API directe
 Exemples introductifsAfficher le code source de la page
Exemples introductifs
Les exemples présentés sur cette page peuvent vous aider à apprendre comment créer des objets avec build123d et constituent un aperçu général de build123d.

Elles sont organisées du plus simple au plus complexe ; les étudier dans l'ordre est donc la meilleure façon de les assimiler.

Note

Certaines lignes importantes sont omises ci-dessous par souci de concision ; vous devrez donc probablement ajouter les lignes 1 et 2 au code fourni ci-dessous pour qu’il fonctionne :

from build123d import *

Si vous utilisez le mode constructeur ou le mode algèbre de build123d ,

Dans ocp_vscode, utilisez simplement, par exemple, la fonction `map` show(ex15)à la fin de votre conception pour visualiser les pièces, les esquisses et les courbes. Cette show_all()fonction permet d'afficher automatiquement tous les objets avec leurs noms de variables sous forme d'étiquettes.

Dans l'éditeur CQ, ajoutez par exemple `<part>` show_object(ex15.part), `<select>` show_object(ex15.sketch)ou show_object(ex15.line)`<line>` à la fin de votre conception pour afficher les pièces, les croquis ou les lignes.

Si vous souhaitez enregistrer votre objet résultant au format STL à partir du mode constructeur , vous pouvez utiliser par exemple .export_stl(ex15.part, "file.stl")

Si vous souhaitez enregistrer votre objet résultant au format STL en mode algébrique , vous pouvez utiliser par exemple :export_stl(ex15, "file.stl")

build123d prend également en charge l'exportation vers de nombreux autres formats de fichiers, notamment STEP ; consultez la page Formats d'import/export pour plus d'informations.

Liste d'exemples

Exemples introductifs

1. Plaque rectangulaire simple

2. Plaque avec trou

3. Un solide prismatique extrudé

4. Construction de profils à l'aide de lignes et d'arcs

5. Déplacement du point de travail actuel

6. Utilisation des listes de points

7. Polygones

8. Polylignes

9. Sélecteurs, filets et chanfreins

10. Sélectionnez Dernier et Trou

11. Utilisez une face comme plan pour BuildSketch et introduisez GridLocations.

12. Définition d'un bord avec une spline

13. Contre-trous, contre-trous et emplacements polaires

14. Positionnez-vous sur une ligne avec '@', '%' et introduisez le balayage

15. Géométrie symétrique par réflexion

16. Mise en miroir d'objets 3D

17. Effet miroir sur les visages

18. Création de placements sur les visages

19. Localisation d'un placement sur un sommet

20. Placement décalé de l'esquisse

21. Créez un plan au centre d'une autre forme.

22. Placements rotatifs

23. Tourner

24. Loft

25. Croquis décalé

26. Décaler une pièce pour créer des éléments fins

27. Division d'un objet

28. Localisation des caractéristiques à partir des visages

29. La bouteille classique OCC

30. Courbe de Bézier

31. Lieux de nidification

32. Boucle for en Python

33. Fonction Python et boucle for

34. Texte en relief et en creux

35. Machines à sous

36. Extruder jusqu'à

37. Positionnement des croquis dans un plan

1. Plaque rectangulaire simple
Un exemple des plus simples, un rectangle Box.

_images/general_ex1.svg
Mode constructeur

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex1:
    Box(length, width, thickness)
Mode algèbre

length, width, thickness = 80.0, 60.0, 10.0

ex1 = Box(length, width, thickness)
2. Plaque avec trou
Une boîte rectangulaire, mais avec un trou en plus.

_images/general_ex2.svg
Mode constructeur

In this case we are using Mode .SUBTRACT to cut the Cylinder from the Box.

length, width, thickness = 80.0, 60.0, 10.0
center_hole_dia = 22.0

with BuildPart() as ex2:
    Box(length, width, thickness)
    Cylinder(radius=center_hole_dia / 2, height=thickness, mode=Mode.SUBTRACT)
Algebra mode

In this case we are using the subtract operator - to cut the Cylinder from the Box.

length, width, thickness = 80.0, 60.0, 10.0
center_hole_dia = 22.0

ex2 = Box(length, width, thickness)
ex2 -= Cylinder(center_hole_dia / 2, height=thickness)
3. An extruded prismatic solid
Build a prismatic solid using extrusion.

_images/general_ex3.svg
Builder mode

This time we can first create a 2D BuildSketch adding a Circle and a subtracted Rectangle and then use BuildPart’s extrude() feature.

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex3:
    with BuildSketch() as ex3_sk:
        Circle(width)
        Rectangle(length / 2, width / 2, mode=Mode.SUBTRACT)
    extrude(amount=2 * thickness)
Algebra mode

This time we can first create a 2D Circle with a subtracted Rectangle` and then use the extrude() operation for parts.

length, width, thickness = 80.0, 60.0, 10.0

sk3 = Circle(width) - Rectangle(length / 2, width / 2)
ex3 = extrude(sk3, amount=2 * thickness)
4. Building Profiles using lines and arcs
Sometimes you need to build complex profiles using lines and arcs. This example builds a prismatic solid from 2D operations. It is not necessary to create variables for the line segments, but it will be useful in a later example.

_images/general_ex4.svg
Builder mode

BuildSketch operates on closed Faces, and the operation make_face() is used to convert the pending line segments from BuildLine into a closed Face.

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex4:
    with BuildSketch() as ex4_sk:
        with BuildLine() as ex4_ln:
            l1 = Line((0, 0), (length, 0))
            l2 = Line((length, 0), (length, width))
            l3 = ThreePointArc((length, width), (width, width * 1.5), (0.0, width))
            l4 = Line((0.0, width), (0, 0))
        make_face()
    extrude(amount=thickness)
Algebra mode

We start with an empty Curve and add lines to it (note that Curve() + [line1, line2, line3] is much more efficient than line1 + line2 + line3, see Performance considerations in algebra mode). The operation make_face() is used to convert the line segments into a Face.

length, width, thickness = 80.0, 60.0, 10.0

lines = Curve() + [
    Line((0, 0), (length, 0)),
    Line((length, 0), (length, width)),
    ThreePointArc((length, width), (width, width * 1.5), (0.0, width)),
    Line((0.0, width), (0, 0)),
]
sk4 = make_face(lines)
ex4 = extrude(sk4, thickness)
Note that to build a closed face it requires line segments that form a closed shape.

5. Moving the current working point
_images/general_ex5.svg
Builder mode

Using Locations we can place one (or multiple) objects at one (or multiple) places.

a, b, c, d = 90, 45, 15, 7.5

with BuildPart() as ex5:
    with BuildSketch() as ex5_sk:
        Circle(a)
        with Locations((b, 0.0)):
            Rectangle(c, c, mode=Mode.SUBTRACT)
        with Locations((0, b)):
            Circle(d, mode=Mode.SUBTRACT)
    extrude(amount=c)
Algebra mode

Using the pattern Pos(x, y, z=0) * obj (with geometry.Pos) we can move an object to the provided position. Using Rot(x_angle, y_angle, z_angle) * obj (with geometry.Rot) would rotate the object.

a, b, c, d = 90, 45, 15, 7.5

sk5 = Circle(a) - Pos(b, 0.0) * Rectangle(c, c) - Pos(0.0, b) * Circle(d)
ex5 = extrude(sk5, c)
6. Using Point Lists
Sometimes you need to create a number of features at various Locations.

_images/general_ex6.svg
Builder mode

You can use a list of points to construct multiple objects at once.

a, b, c = 80, 60, 10

with BuildPart() as ex6:
    with BuildSketch() as ex6_sk:
        Circle(a)
        with Locations((b, 0), (0, b), (-b, 0), (0, -b)):
            Circle(c, mode=Mode.SUBTRACT)
    extrude(amount=c)
Algebra mode

You can use loops to iterate over these Locations or list comprehensions as in the example.

The algebra operations are vectorized, which means obj - [obj1, obj2, obj3] is short for obj - obj1 - obj2 - ob3 (and more efficient, see Performance considerations in algebra mode).

a, b, c = 80, 60, 10

sk6 = [loc * Circle(c) for loc in Locations((b, 0), (0, b), (-b, 0), (0, -b))]
ex6 = extrude(Circle(a) - sk6, c)
7. Polygons
_images/general_ex7.svg
Builder mode

You can create RegularPolygon for each stack point if you would like.

a, b, c = 60, 80, 5

with BuildPart() as ex7:
    with BuildSketch() as ex7_sk:
        Rectangle(a, b)
        with Locations((0, 3 * c), (0, -3 * c)):
            RegularPolygon(radius=2 * c, side_count=6, mode=Mode.SUBTRACT)
    extrude(amount=c)
Algebra mode

You can apply locations to RegularPolygon instances for each location via loops or list comprehensions.

a, b, c = 60, 80, 5

polygons = [
    loc * RegularPolygon(radius=2 * c, side_count=6)
    for loc in Locations((0, 3 * c), (0, -3 * c))
]
sk7 = Rectangle(a, b) - polygons
ex7 = extrude(sk7, amount=c)
8. Polylines
Polyline allows creating a shape from a large number of chained points connected by lines. This example uses a polyline to create one half of an i-beam shape, which is mirror() ed to create the final profile.

_images/general_ex8.svg
Builder mode

(L, H, W, t) = (100.0, 20.0, 20.0, 1.0)
pts = [
    (0, H / 2.0),
    (W / 2.0, H / 2.0),
    (W / 2.0, (H / 2.0 - t)),
    (t / 2.0, (H / 2.0 - t)),
    (t / 2.0, (t - H / 2.0)),
    (W / 2.0, (t - H / 2.0)),
    (W / 2.0, H / -2.0),
    (0, H / -2.0),
]

with BuildPart() as ex8:
    with BuildSketch(Plane.YZ) as ex8_sk:
        with BuildLine() as ex8_ln:
            Polyline(pts)
            mirror(ex8_ln.line, about=Plane.YZ)
        make_face()
    extrude(amount=L)
Algebra mode

(L, H, W, t) = (100.0, 20.0, 20.0, 1.0)
pts = [
    (0, H / 2.0),
    (W / 2.0, H / 2.0),
    (W / 2.0, (H / 2.0 - t)),
    (t / 2.0, (H / 2.0 - t)),
    (t / 2.0, (t - H / 2.0)),
    (W / 2.0, (t - H / 2.0)),
    (W / 2.0, H / -2.0),
    (0, H / -2.0),
]

ln = Polyline(pts)
ln += mirror(ln, Plane.YZ)

sk8 = make_face(Plane.YZ * ln)
ex8 = extrude(sk8, -L).clean()
9. Selectors, Fillets, and Chamfers
This example introduces multiple useful and important concepts. Firstly chamfer() and fillet() can be used to “bevel” and “round” edges respectively. Secondly, these two methods require an edge or a list of edges to operate on. To select all edges, you could simply pass in ex9.edges().

_images/general_ex9.svg
Builder mode

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex9:
    Box(length, width, thickness)
    chamfer(ex9.edges().group_by(Axis.Z)[-1], length=4)
    fillet(ex9.edges().filter_by(Axis.Z), radius=5)
Algebra mode

length, width, thickness = 80.0, 60.0, 10.0

ex9 = Part() + Box(length, width, thickness)
ex9 = chamfer(ex9.edges().group_by(Axis.Z)[-1], length=4)
ex9 = fillet(ex9.edges().filter_by(Axis.Z), radius=5)
Note that group_by() (Axis.Z) returns a list of lists of edges that is grouped by their z-position. In this case we want to use the [-1] group which, by convention, will be the highest z-dimension group.

10. Select Last and Hole
_images/general_ex10.svg
Builder mode

Using Select .LAST you can select the most recently modified edges. It is used to perform a fillet() in this example. This example also makes use of Hole which automatically cuts through the entire part.

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex10:
    Box(length, width, thickness)
    Hole(radius=width / 4)
    fillet(ex10.edges(Select.LAST).group_by(Axis.Z)[-1], radius=2)
Algebra mode

Using the pattern snapshot = obj.edges() before and last_edges = obj.edges() - snapshot after an operation allows to select the most recently modified edges (same for faces, vertices, …). It is used to perform a fillet() in this example. This example also makes use of Hole. Different to the context mode, you have to add the depth of the whole.

ex10 = Part() + Box(length, width, thickness)

snapshot = ex10.edges()
ex10 -= Hole(radius=width / 4, depth=thickness)
last_edges = ex10.edges() - snapshot
ex10 = fillet(last_edges.group_by(Axis.Z)[-1], 2)
11. Use a face as a plane for BuildSketch and introduce GridLocations
_images/general_ex11.svg
Builder mode

BuildSketch accepts a Plane or a Face, so in this case we locate the Sketch on the top of the part. Note that the face used as input to BuildSketch needs to be Planar or unpredictable behavior can result. Additionally GridLocations can be used to create a grid of points that are simultaneously used to place 4 pentagons.

Lastly, extrude() can be used with a negative amount and Mode.SUBTRACT to cut these from the parent.

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex11:
    Box(length, width, thickness)
    chamfer(ex11.edges().group_by(Axis.Z)[-1], length=4)
    fillet(ex11.edges().filter_by(Axis.Z), radius=5)
    Hole(radius=width / 4)
    fillet(ex11.edges(Select.LAST).sort_by(Axis.Z)[-1], radius=2)
    with BuildSketch(ex11.faces().sort_by(Axis.Z)[-1]) as ex11_sk:
        with GridLocations(length / 2, width / 2, 2, 2):
            RegularPolygon(radius=5, side_count=5)
    extrude(amount=-thickness, mode=Mode.SUBTRACT)
Algebra mode

The pattern plane * obj can be used to locate an object on a plane. Furthermore, the pattern plane * location * obj first places the object on a plane and then moves it relative to plane according to location.

GridLocations creates a grid of points that can be used in loops or list comprehensions as described earlier.

Lastly, extrude() can be used with a negative amount and cut (-) from the parent.

length, width, thickness = 80.0, 60.0, 10.0

ex11 = Part() + Box(length, width, thickness)
ex11 = chamfer(ex11.edges().group_by()[-1], 4)
ex11 = fillet(ex11.edges().filter_by(Axis.Z), 5)
last = ex11.edges()
ex11 -= Hole(radius=width / 4, depth=thickness)
ex11 = fillet((ex11.edges() - last).sort_by().last, 2)

plane = Plane(ex11.faces().sort_by().last)
polygons = Sketch() + [
    plane * loc * RegularPolygon(radius=5, side_count=5)
    for loc in GridLocations(length / 2, width / 2, 2, 2)
]
ex11 -= extrude(polygons, -thickness)
Note that the direction implied by positive or negative inputs to amount is relative to the normal direction of the face or plane. As a result of this, unexpected behavior can occur if the extrude direction and mode/operation (ADD / + or SUBTRACT / -) are not correctly set.

12. Defining an Edge with a Spline
This example defines a side using a spline curve through a collection of points. Useful when you have an edge that needs a complex profile.

_images/general_ex12.svg
Builder mode

pts = [
    (55, 30),
    (50, 35),
    (40, 30),
    (30, 20),
    (20, 25),
    (10, 20),
    (0, 20),
]

with BuildPart() as ex12:
    with BuildSketch() as ex12_sk:
        with BuildLine() as ex12_ln:
            l1 = Spline(pts)
            l2 = Line((55, 30), (60, 0))
            l3 = Line((60, 0), (0, 0))
            l4 = Line((0, 0), (0, 20))
        make_face()
    extrude(amount=10)
Algebra mode

pts = [
    (55, 30),
    (50, 35),
    (40, 30),
    (30, 20),
    (20, 25),
    (10, 20),
    (0, 20),
]

l1 = Spline(pts)
l2 = Line(l1 @ 0, (60, 0))
l3 = Line(l2 @ 1, (0, 0))
l4 = Line(l3 @ 1, l1 @ 1)

sk12 = make_face([l1, l2, l3, l4])
ex12 = extrude(sk12, 10)
13. CounterBoreHoles, CounterSinkHoles, and PolarLocations
Counter-sink and counter-bore holes are useful for creating recessed areas for fasteners.

_images/general_ex13.svg
Builder mode

We use a face to establish a location for Locations.

a, b = 40, 4
with BuildPart() as ex13:
    Cylinder(radius=50, height=10)
    with Locations(ex13.faces().sort_by(Axis.Z)[-1]):
        with PolarLocations(radius=a, count=4):
            CounterSinkHole(radius=b, counter_sink_radius=2 * b)
        with PolarLocations(radius=a, count=4, start_angle=45, angular_range=360):
            CounterBoreHole(radius=b, counter_bore_radius=2 * b, counter_bore_depth=b)
Algebra mode

We use a face to establish a plane that is used later in the code for locating objects onto this plane.

a, b = 40, 4

ex13 = Cylinder(radius=50, height=10)
plane = Plane(ex13.faces().sort_by().last)

ex13 -= (
    plane
    * PolarLocations(radius=a, count=4)
    * CounterSinkHole(radius=b, counter_sink_radius=2 * b, depth=10)
)
ex13 -= (
    plane
    * PolarLocations(radius=a, count=4, start_angle=45, angular_range=360)
    * CounterBoreHole(
        radius=b, counter_bore_radius=2 * b, depth=10, counter_bore_depth=b
    )
)
PolarLocations creates a list of points that are radially distributed.

14. Position on a line with ‘@’, ‘%’ and introduce Sweep
build123d includes a feature for finding the position along a line segment. This is normalized between 0 and 1 and can be accessed using the position_at() (@) operator. Similarly the tangent_at() (%) operator returns the line direction at a given point.

These two features are very powerful for chaining line segments together without having to repeat dimensions again and again, which is error prone, time consuming, and more difficult to maintain. The pending faces must lie on the path, please see example 37 for a way to make this placement easier.

_images/general_ex14.svg
Builder mode

The sweep() method takes any pending faces and sweeps them through the provided path (in this case the path is taken from the pending edges from ex14_ln). revolve() requires a single connected wire.

a, b = 40, 20

with BuildPart() as ex14:
    with BuildLine() as ex14_ln:
        l1 = JernArc(start=(0, 0), tangent=(0, 1), radius=a, arc_size=180)
        l2 = JernArc(start=l1 @ 1, tangent=l1 % 1, radius=a, arc_size=-90)
        l3 = Line(l2 @ 1, l2 @ 1 + (-a, a))
    with BuildSketch(Plane.XZ) as ex14_sk:
        Rectangle(b, b)
    sweep()
Algebra mode

The sweep() method takes any faces and sweeps them through the provided path (in this case the path is taken from ex14_ln).

a, b = 40, 20

l1 = JernArc(start=(0, 0), tangent=(0, 1), radius=a, arc_size=180)
l2 = JernArc(start=l1 @ 1, tangent=l1 % 1, radius=a, arc_size=-90)
l3 = Line(l2 @ 1, l2 @ 1 + (-a, a))
ex14_ln = l1 + l2 + l3

sk14 = Plane.XZ * Rectangle(b, b)
ex14 = sweep(sk14, path=ex14_ln)
It is also possible to use tuple or Vector addition (and other vector math operations) as seen in the l3 variable.

15. Mirroring Symmetric Geometry
Here mirror is used on the BuildLine to create a symmetric shape with fewer line segment commands. Additionally the ‘@’ operator is used to simplify the line segment commands.

(l4 @ 1).Y is used to extract the y-component of the l4 @ 1 vector.

_images/general_ex15.svg
Builder mode

a, b, c = 80, 40, 20

with BuildPart() as ex15:
    with BuildSketch() as ex15_sk:
        with BuildLine() as ex15_ln:
            l1 = Line((0, 0), (a, 0))
            l2 = Line(l1 @ 1, l1 @ 1 + (0, b))
            l3 = Line(l2 @ 1, l2 @ 1 + (-c, 0))
            l4 = Line(l3 @ 1, l3 @ 1 + (0, -c))
            l5 = Line(l4 @ 1, (0, (l4 @ 1).Y))
            mirror(ex15_ln.line, about=Plane.YZ)
        make_face()
    extrude(amount=c)
Algebra mode

Combine lines via the pattern Curve() + [l1, l2, l3, l4, l5]

a, b, c = 80, 40, 20

l1 = Line((0, 0), (a, 0))
l2 = Line(l1 @ 1, l1 @ 1 + (0, b))
l3 = Line(l2 @ 1, l2 @ 1 + (-c, 0))
l4 = Line(l3 @ 1, l3 @ 1 + (0, -c))
l5 = Line(l4 @ 1, (0, (l4 @ 1).Y))
ln = Curve() + [l1, l2, l3, l4, l5]
ln += mirror(ln, Plane.YZ)

sk15 = make_face(ln)
ex15 = extrude(sk15, c)
16. Mirroring 3D Objects
Mirror can also be used with BuildPart (and BuildSketch) to mirror 3D objects. The Plane.offset() method shifts the plane in the normal direction (positive or negative).

_images/general_ex16.svg
Builder mode

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex16_single:
    with BuildSketch(Plane.XZ) as ex16_sk:
        Rectangle(length, width)
        fillet(ex16_sk.vertices(), radius=length / 10)
        with GridLocations(x_spacing=length / 4, y_spacing=0, x_count=3, y_count=1):
            Circle(length / 12, mode=Mode.SUBTRACT)
        Rectangle(length, width, align=(Align.MIN, Align.MIN), mode=Mode.SUBTRACT)
    extrude(amount=length)

with BuildPart() as ex16:
    add(ex16_single.part)
    mirror(ex16_single.part, about=Plane.XY.offset(width))
    mirror(ex16_single.part, about=Plane.YX.offset(width))
    mirror(ex16_single.part, about=Plane.YZ.offset(width))
    mirror(ex16_single.part, about=Plane.YZ.offset(-width))
Algebra mode

length, width, thickness = 80.0, 60.0, 10.0

sk16 = Rectangle(length, width)
sk16 = fillet(sk16.vertices(), length / 10)

circles = [loc * Circle(length / 12) for loc in GridLocations(length / 4, 0, 3, 1)]

sk16 = sk16 - circles - Rectangle(length, width, align=(Align.MIN, Align.MIN))
ex16_single = extrude(Plane.XZ * sk16, length)

planes = [
    Plane.XY.offset(width),
    Plane.YX.offset(width),
    Plane.YZ.offset(width),
    Plane.YZ.offset(-width),
]
objs = [mirror(ex16_single, plane) for plane in planes]
ex16 = ex16_single + objs
17. Mirroring From Faces
Here we select the farthest face in the Y-direction and turn it into a Plane using the Plane() class.

_images/general_ex17.svg
Builder mode

a, b = 30, 20

with BuildPart() as ex17:
    with BuildSketch() as ex17_sk:
        RegularPolygon(radius=a, side_count=5)
    extrude(amount=b)
    mirror(ex17.part, about=Plane(ex17.faces().group_by(Axis.Y)[0][0]))
Algebra mode

a, b = 30, 20

sk17 = RegularPolygon(radius=a, side_count=5)
ex17 = extrude(sk17, amount=b)
ex17 += mirror(ex17, Plane(ex17.faces().sort_by(Axis.Y).first))
18. Creating Placements on Faces
Here we start with an earlier example, select the top face, draw a rectangle and then use Extrude with a negative distance.

_images/general_ex18.svg
Builder mode

We then use Mode.SUBTRACT to cut it out from the main body.

length, width, thickness = 80.0, 60.0, 10.0
a, b = 4, 5

with BuildPart() as ex18:
    Box(length, width, thickness)
    chamfer(ex18.edges().group_by(Axis.Z)[-1], length=a)
    fillet(ex18.edges().filter_by(Axis.Z), radius=b)
    with BuildSketch(ex18.faces().sort_by(Axis.Z)[-1]):
        Rectangle(2 * b, 2 * b)
    extrude(amount=-thickness, mode=Mode.SUBTRACT)
Algebra mode

We then use -= to cut it out from the main body.

length, width, thickness = 80.0, 60.0, 10.0
a, b = 4, 5

ex18 = Part() + Box(length, width, thickness)
ex18 = chamfer(ex18.edges().group_by()[-1], a)
ex18 = fillet(ex18.edges().filter_by(Axis.Z), b)

sk18 = Plane(ex18.faces().sort_by().first) * Rectangle(2 * b, 2 * b)
ex18 -= extrude(sk18, -thickness)
19. Locating a placement on a vertex
Here a face is selected and two different strategies are used to select vertices. Firstly vtx uses group_by() and Axis.X to select a particular vertex. The second strategy uses a custom defined Axis vtx2Axis that is pointing roughly in the direction of a vertex to select, and then sort_by() this custom Axis.

_images/general_ex19.svg
Builder mode

Then the X and Y positions of these vertices are selected and passed to Locations as center points for two circles that cut through the main part. Note that if you passed the variable vtx directly to Locations then the part would be offset from the sketch placement by the vertex z-position.

length, thickness = 80.0, 10.0

with BuildPart() as ex19:
    with BuildSketch() as ex19_sk:
        RegularPolygon(radius=length / 2, side_count=7)
    extrude(amount=thickness)
    topf = ex19.faces().sort_by(Axis.Z)[-1]
    vtx = topf.vertices().group_by(Axis.X)[-1][0]
    vtx2Axis = Axis((0, 0, 0), (-1, -0.5, 0))
    vtx2 = topf.vertices().sort_by(vtx2Axis)[-1]
    with BuildSketch(topf) as ex19_sk2:
        with Locations((vtx.X, vtx.Y), (vtx2.X, vtx2.Y)):
            Circle(radius=length / 8)
    extrude(amount=-thickness, mode=Mode.SUBTRACT)
Algebra mode

Then the X and Y positions of these vertices are selected and used to move two circles that cut through the main part. Note that if you passed the variable vtx directly to Pos then the part would be offset from the sketch placement by the vertex z-position.

length, thickness = 80.0, 10.0

ex19_sk = RegularPolygon(radius=length / 2, side_count=7)
ex19 = extrude(ex19_sk, thickness)

topf = ex19.faces().sort_by().last

vtx = topf.vertices().group_by(Axis.X)[-1][0]

vtx2Axis = Axis((0, 0, 0), (-1, -0.5, 0))
vtx2 = topf.vertices().sort_by(vtx2Axis)[-1]

ex19_sk2 = Circle(radius=length / 8)
ex19_sk2 = Pos(vtx.X, vtx.Y) * ex19_sk2 + Pos(vtx2.X, vtx2.Y) * ex19_sk2

ex19 -= extrude(ex19_sk2, thickness)
20. Offset Sketch Placement
The plane variable is set to be coincident with the farthest face in the negative x-direction. The resulting Plane is offset from the original position.

_images/general_ex20.svg
Builder mode

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex20:
    Box(length, width, thickness)
    plane = Plane(ex20.faces().group_by(Axis.X)[0][0])
    with BuildSketch(plane.offset(2 * thickness)):
        Circle(width / 3)
    extrude(amount=width)
Algebra mode

length, width, thickness = 80.0, 60.0, 10.0

ex20 = Box(length, width, thickness)
plane = Plane(ex20.faces().sort_by(Axis.X).first).offset(2 * thickness)

sk20 = plane * Circle(width / 3)
ex20 += extrude(sk20, width)
21. Create a Plane in the center of another shape
On crée un cylindre, puis on utilise l'origine et la direction z de cette partie pour créer un nouveau plan permettant de positionner un autre cylindre perpendiculairement et à mi-chemin du premier.

_images/general_ex21.svg
Mode constructeur

width, length = 10.0, 60.0

with BuildPart() as ex21:
    with BuildSketch() as ex21_sk:
        Circle(width / 2)
    extrude(amount=length)
    with BuildSketch(Plane(origin=ex21.part.center(), z_dir=(-1, 0, 0))):
        Circle(width / 2)
    extrude(amount=length)
Mode algèbre

width, length = 10.0, 60.0

ex21 = extrude(Circle(width / 2), length)
plane = Plane(origin=ex21.center(), z_dir=(-1, 0, 0))
ex21 += plane * extrude(Circle(width / 2), length)
22. Placements rotatifs
Il est également possible de créer un placement d'esquisse pivoté, en s'appuyant sur certains concepts d'un exemple précédent.

_images/general_ex22.svg
Mode constructeur

Utilisez la rotated()méthode pour faire pivoter le plan de placement.

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex22:
    Box(length, width, thickness)
    pln = Plane(ex22.faces().group_by(Axis.Z)[0][0]).rotated((0, -50, 0))
    with BuildSketch(pln) as ex22_sk:
        with GridLocations(length / 4, width / 4, 2, 2):
            Circle(thickness / 4)
    extrude(amount=-100, both=True, mode=Mode.SUBTRACT)
Mode algèbre

Utilisez l'opérateur *pour déplacer le plan (après multiplication !).

length, width, thickness = 80.0, 60.0, 10.0

ex22 = Box(length, width, thickness)
plane = Plane((ex22.faces().group_by(Axis.Z)[0])[0]) * Rot(0, 50, 0)

holes = Sketch() + [
    plane * loc * Circle(thickness / 4)
    for loc in GridLocations(length / 4, width / 4, 2, 2)
]
ex22 -= extrude(holes, -100, both=True)
GridLocationsplace 4 cercles sur 4 points de cette disposition pivotée, puis les cercles sont extrudés dans la direction normale « à la fois » (positive et négative).

23. Tourner
Ici, nous créons une esquisse avec un axe de rotation Polyline, un axe de rotation Lineet un axe de Circlerévolution. Il est absolument essentiel que l'esquisse ne se trouve que d'un seul côté de l'axe de rotation avant l'appel à la fonction Revolve. À cette fin, spliton utilise l'option `--split` avec `--split` Plane.ZYpour ne conserver qu'un seul côté de l'esquisse.

Il est fortement recommandé de visualiser votre croquis avant de tenter d'appeler la fonction de révolution.

_images/general_ex23.svg
Mode constructeur

pts = [
    (-25, 35),
    (-25, 0),
    (-20, 0),
    (-20, 5),
    (-15, 10),
    (-15, 35),
]

with BuildPart() as ex23:
    with BuildSketch(Plane.XZ) as ex23_sk:
        with BuildLine() as ex23_ln:
            l1 = Polyline(pts)
            l2 = Line(l1 @ 1, l1 @ 0)
        make_face()
        with Locations((0, 35)):
            Circle(25)
        split(bisect_by=Plane.ZY)
    revolve(axis=Axis.Z)
Mode algèbre

pts = [
    (-25, 35),
    (-25, 0),
    (-20, 0),
    (-20, 5),
    (-15, 10),
    (-15, 35),
]

l1 = Polyline(pts)
l2 = Line(l1 @ 1, l1 @ 0)
sk23 = make_face([l1, l2])

sk23 += Pos(0, 35) * Circle(25)
sk23 = Plane.XZ * split(sk23, bisect_by=Plane.ZY)

ex23 = revolve(sk23, Axis.Z)
24. Loft
L'outil Loft est très puissant pour assembler des formes dissemblables. Ici, nous créons une forme conique à partir d'un cercle et d'un rectangle décalé verticalement. Loft loft()prend automatiquement en compte les faces ajoutées par les deux BuildSketches. Attention : le comportement de Loft peut être imprévisible si les faces d'entrée ne sont pas parallèles.

_images/general_ex24.svg
Mode constructeur

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex24:
    Box(length, length, thickness)
    with BuildSketch(ex24.faces().group_by(Axis.Z)[0][0]) as ex24_sk:
        Circle(length / 3)
    with BuildSketch(ex24_sk.faces()[0].offset(length / 2)) as ex24_sk2:
        Rectangle(length / 6, width / 6)
    loft()
Mode algèbre

length, width, thickness = 80.0, 60.0, 10.0

ex24 = Box(length, length, thickness)
plane = Plane(ex24.faces().sort_by().last)

faces = Sketch() + [
    plane * Circle(length / 3),
    plane.offset(length / 2) * Rectangle(length / 6, width / 6),
]

ex24 += loft(faces)
25. Croquis décalé
_images/general_ex25.svg
Mode constructeur

Les faces de BuildSketch peuvent être transformées avec un 2D offset().

rad, offs = 50, 10

with BuildPart() as ex25:
    with BuildSketch() as ex25_sk1:
        RegularPolygon(radius=rad, side_count=5)
    with BuildSketch(Plane.XY.offset(15)) as ex25_sk2:
        RegularPolygon(radius=rad, side_count=5)
        offset(amount=offs)
    with BuildSketch(Plane.XY.offset(30)) as ex25_sk3:
        RegularPolygon(radius=rad, side_count=5)
        offset(amount=offs, kind=Kind.INTERSECTION)
    extrude(amount=1)
Mode algèbre

Les visages de croquis peuvent être transformés avec un 2D offset().

rad, offs = 50, 10

sk25_1 = RegularPolygon(radius=rad, side_count=5)
sk25_2 = Plane.XY.offset(15) * RegularPolygon(radius=rad, side_count=5)
sk25_2 = offset(sk25_2, offs)
sk25_3 = Plane.XY.offset(30) * RegularPolygon(radius=rad, side_count=5)
sk25_3 = offset(sk25_3, offs, kind=Kind.INTERSECTION)

sk25 = Sketch() + [sk25_1, sk25_2, sk25_3]
ex25 = extrude(sk25, 1)
Ils peuvent être décalés vers l'intérieur ou vers l'extérieur, et avec différentes techniques pour étendre les coins (voir Kinddans la documentation sur le décalage).

26. Décaler une pièce pour créer des éléments fins
Les pièces peuvent également être transformées à l'aide d'un décalage, mais dans ce cas avec une structure 3D offset(). Aussi appelée coque, cette technique permet de créer des parois fines en un minimum d'opérations. Le décalage peut être effectué vers l'intérieur ou vers l'extérieur. Les faces à supprimer peuvent être sélectionnées à l'aide du openingsparamètre correspondant offset().

Notez que les arêtes et/ou faces auto-intersectantes peuvent rompre les décalages 2D et 3D.

_images/general_ex26.svg
Mode constructeur

length, width, thickness, wall = 80.0, 60.0, 10.0, 2.0

with BuildPart() as ex26:
    Box(length, width, thickness)
    topf = ex26.faces().sort_by(Axis.Z)[-1]
    offset(amount=-wall, openings=topf)
Mode algèbre

length, width, thickness, wall = 80.0, 60.0, 10.0, 2.0

ex26 = Box(length, width, thickness)
topf = ex26.faces().sort_by().last
ex26 = offset(ex26, amount=-wall, openings=topf)
27. Division d'un objet
Vous pouvez diviser un objet à l'aide d'un plan et conserver une ou les deux moitiés. Ici, nous sélectionnons une face et la décalons de la moitié de la largeur du cube.

_images/general_ex27.svg
Mode constructeur

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex27:
    Box(length, width, thickness)
    with BuildSketch(ex27.faces().sort_by(Axis.Z)[0]) as ex27_sk:
        Circle(width / 4)
    extrude(amount=-thickness, mode=Mode.SUBTRACT)
    split(bisect_by=Plane(ex27.faces().sort_by(Axis.Y)[-1]).offset(-width / 2))
Mode algèbre

length, width, thickness = 80.0, 60.0, 10.0

ex27 = Box(length, width, thickness)
sk27 = Plane(ex27.faces().sort_by().first) * Circle(width / 4)
ex27 -= extrude(sk27, -thickness)
ex27 = split(ex27, Plane(ex27.faces().sort_by(Axis.Y).last).offset(-width / 2))
28. Localisation des caractéristiques à partir des visages
_images/general_ex28.svg
Mode constructeur

Nous créons un prisme triangulaire , puis nous utilisons les faces de cet objet pour découper des trous dans une sphère.Mode .PRIVATE

width, thickness = 80.0, 10.0

with BuildPart() as ex28:
    with BuildSketch() as ex28_sk:
        RegularPolygon(radius=width / 4, side_count=3)
    ex28_ex = extrude(amount=thickness, mode=Mode.PRIVATE)
    midfaces = ex28_ex.faces().group_by(Axis.Z)[1]
    Sphere(radius=width / 2)
    for face in midfaces:
        with Locations(face):
            Hole(thickness / 2)
Mode algèbre

On crée un prisme triangulaire, puis on utilise les faces de cet objet pour découper des trous dans une sphère.

width, thickness = 80.0, 10.0

sk28 = RegularPolygon(radius=width / 4, side_count=3)
tmp28 = extrude(sk28, thickness)
ex28 = Sphere(radius=width / 2)
for p in [Plane(face) for face in tmp28.faces().group_by(Axis.Z)[1]]:
    ex28 -= p * Hole(thickness / 2, depth=width)
Nous pouvons créer plusieurs plans de placement en parcourant la liste des faces.

29. La bouteille classique OCC
build123d est basé sur le noyau de modélisation OpenCascade.org (OCC). Les connaisseurs d'OCC connaissent le célèbre exemple de la « bouteille ». Nous utilisons un décalage 3D et le paramètre d'ouverture pour créer l'ouverture de la bouteille.

_images/general_ex29.svg
Mode constructeur

L, w, t, b, h, n = 60.0, 18.0, 9.0, 0.9, 90.0, 6.0

with BuildPart() as ex29:
    with BuildSketch(Plane.XY.offset(-b)) as ex29_ow_sk:
        with BuildLine() as ex29_ow_ln:
            l1 = Line((0, 0), (0, w / 2))
            l2 = ThreePointArc(l1 @ 1, (L / 2.0, w / 2.0 + t), (L, w / 2.0))
            l3 = Line(l2 @ 1, ((l2 @ 1).X, 0, 0))
            mirror(ex29_ow_ln.line)
        make_face()
    extrude(amount=h + b)
    fillet(ex29.edges(), radius=w / 6)
    with BuildSketch(ex29.faces().sort_by(Axis.Z)[-1]):
        Circle(t)
    extrude(amount=n)
    necktopf = ex29.faces().sort_by(Axis.Z)[-1]
    offset(ex29.solids()[0], amount=-b, openings=necktopf)
Mode algèbre

L, w, t, b, h, n = 60.0, 18.0, 9.0, 0.9, 90.0, 8.0

l1 = Line((0, 0), (0, w / 2))
l2 = ThreePointArc(l1 @ 1, (L / 2.0, w / 2.0 + t), (L, w / 2.0))
l3 = Line(l2 @ 1, ((l2 @ 1).X, 0, 0))
ln29 = l1 + l2 + l3
ln29 += mirror(ln29)
sk29 = make_face(ln29)
ex29 = extrude(sk29, -(h + b))
ex29 = fillet(ex29.edges(), radius=w / 6)

neck = Plane(ex29.faces().sort_by().last) * Circle(t)
ex29 += extrude(neck, n)
necktopf = ex29.faces().sort_by().last
ex29 = offset(ex29, -b, openings=necktopf)
30. Courbe de Bézier
Ici , ptscette valeur sert d'entrée à la fois à l'outil Polylineet à l'outil Bézier seul. Ces deux outils combinés créent une ligne fermée qui est transformée en une face et extrudée.Bezierwts

_images/general_ex30.svg
Mode constructeur

pts = [
    (0, 0),
    (20, 20),
    (40, 0),
    (0, -40),
    (-60, 0),
    (0, 100),
    (100, 0),
]

wts = [
    1.0,
    1.0,
    2.0,
    3.0,
    4.0,
    2.0,
    1.0,
]

with BuildPart() as ex30:
    with BuildSketch() as ex30_sk:
        with BuildLine() as ex30_ln:
            l0 = Polyline(pts)
            l1 = Bezier(pts, weights=wts)
        make_face()
    extrude(amount=10)
Mode algèbre

pts = [
    (0, 0),
    (20, 20),
    (40, 0),
    (0, -40),
    (-60, 0),
    (0, 100),
    (100, 0),
]

wts = [
    1.0,
    1.0,
    2.0,
    3.0,
    4.0,
    2.0,
    1.0,
]

ex30_ln = Polyline(pts) + Bezier(pts, weights=wts)
ex30_sk = make_face(ex30_ln)
ex30 = extrude(ex30_sk, -10)
31. Lieux de nidification
Les contextes de localisation peuvent être imbriqués pour créer des groupes de formes. Ici, 24 triangles, 6 carrés et 1 hexagone sont créés puis extrudés. À noter que PolarLocations les groupes « enfants » sont pivotés par défaut.

_images/general_ex31.svg
Mode constructeur

a, b, c = 80.0, 5.0, 3.0

with BuildPart() as ex31:
    with BuildSketch() as ex31_sk:
        with PolarLocations(a / 2, 6):
            with GridLocations(3 * b, 3 * b, 2, 2):
                RegularPolygon(b, 3)
            RegularPolygon(b, 4)
        RegularPolygon(3 * b, 6, rotation=30)
    extrude(amount=c)
Mode algèbre

a, b, c = 80.0, 5.0, 3.0

ex31 = Rot(Z=30) * RegularPolygon(3 * b, 6)
ex31 += PolarLocations(a / 2, 6) * (
    RegularPolygon(b, 4) + GridLocations(3 * b, 3 * b, 2, 2) * RegularPolygon(b, 3)
)
ex31 = extrude(ex31, 3)
32. Boucle for en Python
Dans cet exemple, une boucle for standard de Python est utilisée avec une liste de faces extraites d'une esquisse pour modifier progressivement le degré d'extrusion. L'esquisse comporte 7 faces, ce qui entraîne 7 appels distincts à la fonction extrude().

_images/general_ex32.svg
Mode constructeur

Mode .PRIVATEest utilisé BuildSketchpour éviter d'ajouter ces faces jusqu'à la fin de la boucle for.

a, b, c = 80.0, 10.0, 1.0

with BuildPart() as ex32:
    with BuildSketch(mode=Mode.PRIVATE) as ex32_sk:
        RegularPolygon(2 * b, 6, rotation=30)
        with PolarLocations(a / 2, 6):
            RegularPolygon(b, 4)
    for idx, obj in enumerate(ex32_sk.sketch.faces()):
        add(obj)
        extrude(amount=c + 3 * idx)
Mode algèbre

a, b, c = 80.0, 10.0, 1.0

ex32_sk = RegularPolygon(2 * b, 6, rotation=30)
ex32_sk += PolarLocations(a / 2, 6) * RegularPolygon(b, 4)
ex32 = Part() + [extrude(obj, c + 3 * idx) for idx, obj in enumerate(ex32_sk.faces())]
33. Fonction Python et boucle for
En reprenant l'exemple précédent, une fonction Python standard est utilisée pour renvoyer un croquis en fonction de plusieurs entrées afin de modifier progressivement la taille de chaque carré.

_images/general_ex33.svg
Mode constructeur

La fonction renvoie un BuildSketch.

a, b, c = 80.0, 5.0, 1.0


def square(rad, loc):
    with BuildSketch() as sk:
        with Locations(loc):
            RegularPolygon(rad, 4)
    return sk.sketch


with BuildPart() as ex33:
    with BuildSketch(mode=Mode.PRIVATE) as ex33_sk:
        locs = PolarLocations(a / 2, 6)
        for i, j in enumerate(locs):
            add(square(b + 2 * i, j))
    for idx, obj in enumerate(ex33_sk.sketch.faces()):
        add(obj)
        extrude(amount=c + 2 * idx)
Mode algèbre

La fonction renvoie un Sketchobjet.

a, b, c = 80.0, 5.0, 1.0


def square(rad, loc):
    return loc * RegularPolygon(rad, 4)


ex33 = Part() + [
    extrude(square(b + 2 * i, loc), c + 2 * i)
    for i, loc in enumerate(PolarLocations(a / 2, 6))
]
34. Texte en relief et en creux
_images/general_ex34.svg
Mode constructeur

Le texte « Hello » est placé sur un rectangle et mis en relief grâce à un objet BuildSketch appliqué sur sa face supérieure topf. Notez que la variable Alignest utilisée pour contrôler le positionnement du texte. Nous réutilisons cette topfvariable pour sélectionner la même face et mettre en creux le texte « World ». Si nous exécutions simplement la commande BuildSketch(ex34.faces().sort_by(Axis.Z)[-1])pour les deux textes , le second texte « World » se retrouverait incorrectement au-dessus du texte « Hello ».ex34_sk1 & 2

length, width, thickness, fontsz, fontht = 80.0, 60.0, 10.0, 25.0, 4.0

with BuildPart() as ex34:
    Box(length, width, thickness)
    topf = ex34.faces().sort_by(Axis.Z)[-1]
    with BuildSketch(topf) as ex34_sk:
        Text("Hello", font_size=fontsz, align=(Align.CENTER, Align.MIN))
    extrude(amount=fontht)
    with BuildSketch(topf) as ex34_sk2:
        Text("World", font_size=fontsz, align=(Align.CENTER, Align.MAX))
    extrude(amount=-fontht, mode=Mode.SUBTRACT)
Mode algèbre

Le texte « Hello » est placé sur un rectangle et mis en relief grâce à un tracé sur sa face supérieure topf. Notez que Aligncette variable permet de contrôler le positionnement du texte. Nous la réutilisons topfpour sélectionner la même face et mettre en creux le texte « World ».

length, width, thickness, fontsz, fontht = 80.0, 60.0, 10.0, 25.0, 4.0

ex34 = Box(length, width, thickness)
plane = Plane(ex34.faces().sort_by().last)
ex34_sk = plane * Text("Hello", font_size=fontsz, align=(Align.CENTER, Align.MIN))
ex34 += extrude(ex34_sk, amount=fontht)
ex34_sk2 = plane * Text("World", font_size=fontsz, align=(Align.CENTER, Align.MAX))
ex34 -= extrude(ex34_sk2, amount=-fontht)
35. Machines à sous
_images/general_ex35.svg
Mode constructeur

Ici, nous créons un SlotCenterToCenteret utilisons ensuite un BuildLineet RadiusArcpour créer un arc pour deux instances de SlotArc.

length, width, thickness = 80.0, 60.0, 10.0

with BuildPart() as ex35:
    Box(length, length, thickness)
    topf = ex35.faces().sort_by(Axis.Z)[-1]
    with BuildSketch(topf) as ex35_sk:
        SlotCenterToCenter(width / 2, 10)
        with BuildLine(mode=Mode.PRIVATE) as ex35_ln:
            RadiusArc((-width / 2, 0), (0, width / 2), radius=width / 2)
        SlotArc(arc=ex35_ln.edges()[0], height=thickness, rotation=0)
        with BuildLine(mode=Mode.PRIVATE) as ex35_ln2:
            RadiusArc((0, -width / 2), (width / 2, 0), radius=-width / 2)
        SlotArc(arc=ex35_ln2.edges()[0], height=thickness, rotation=0)
    extrude(amount=-thickness, mode=Mode.SUBTRACT)
Mode algèbre

Ici, nous créons un SlotCenterToCenteret utilisons ensuite un RadiusArcpour créer un arc pour deux instances de SlotArc.

length, width, thickness = 80.0, 60.0, 10.0

ex35 = Box(length, length, thickness)
plane = Plane(ex35.faces().sort_by().last)
ex35_sk = SlotCenterToCenter(width / 2, 10)
ex35_ln = RadiusArc((-width / 2, 0), (0, width / 2), radius=width / 2)
ex35_sk += SlotArc(arc=ex35_ln.edges()[0], height=thickness)
ex35_ln2 = RadiusArc((0, -width / 2), (width / 2, 0), radius=-width / 2)
ex35_sk += SlotArc(arc=ex35_ln2.edges()[0], height=thickness)
ex35 -= extrude(plane * ex35_sk, -thickness)
36. Extruder jusqu'à
Parfois, vous voudrez extruder jusqu'à une face donnée qui peut ne pas être plane ou pour laquelle vous ne pouvez pas déterminer facilement la distance d'extrusion. Dans de tels cas, vous pouvez utiliser ` with` ou ` .`.extrude() UntilUntil.NEXTUntil.LAST

_images/general_ex36.svg
Mode constructeur

rad, rev = 6, 50

with BuildPart() as ex36:
    with BuildSketch() as ex36_sk:
        with Locations((0, rev)):
            Circle(rad)
    revolve(axis=Axis.X, revolution_arc=180)
    with BuildSketch() as ex36_sk2:
        Rectangle(rad, rev)
    extrude(until=Until.NEXT)
Mode algèbre

rad, rev = 6, 50

ex36_sk = Pos(0, rev) * Circle(rad)
ex36 = revolve(axis=Axis.X, profiles=ex36_sk, revolution_arc=180)
ex36_sk2 = Rectangle(rad, rev)
ex36 += extrude(ex36_sk2, until=Until.NEXT, target=ex36)
37. Positionnement des croquis dans un plan
PlaneOn peut positionner un objet directement à partir de coordonnées ou en déplaçant l'origine d'un plan existant vers une géométrie sélectionnée dans une esquisse précédente. Ici shift_origin(), le cercle est placé au sommet inférieur gauche du rectangle, tandis qu'un second plan positionne l'ellipse à partir de coordonnées explicites.

_images/general_ex37.svg
Mode constructeur

with BuildPart() as ex37:
    with BuildSketch() as ex37_sk:
        Rectangle(1, 2, align=(Align.CENTER, Align.MIN))
    with BuildSketch(
        Plane.XY.shift_origin(ex37_sk.vertices().group_by(Axis.Y)[0].sort_by(Axis.X)[0])
    ):
        Circle(1)
    with BuildSketch(Plane((0.5, 2))):
        Ellipse(0.5, 1)
    extrude(amount=1)
© Copyright 2022, Gumyr.

Construit avec Sphinx en utilisant un thème fourni par Read the Docs .
Lisez les documents
 dernier
Vous n'avez pas besoin d'une base de données distincte pour commencer à développer des applications de génération d'IA. Atlas suffit.
Annonces diffusées par EthicalAds
Fermer l'annonce