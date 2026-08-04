// Backward-facing step geometry (2D profile extruded)
// Reference: OpenFOAM pitzDaily
//
// Profile in XY plane (Z=0):
//   Inlet channel:   0<x<1,   0<y<1.0   (height 1.0m)
//   Step riser:      x=1,     0<y<0.5
//   Outlet channel:  1<x<6,   0.5<y<0.6  (height 0.1m)
// Extruded 0.01 m in Z for 2D simulation

SetFactory("OpenCASCADE");

lc = 0.02;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {1.0, 0.0, 0.0, lc};
Point(3) = {1.0, 0.5, 0.0, lc};
Point(4) = {6.0, 0.5, 0.0, lc};
Point(5) = {6.0, 0.6, 0.0, lc};
Point(6) = {1.0, 0.6, 0.0, lc};
Point(7) = {1.0, 1.0, 0.0, lc};
Point(8) = {0.0, 1.0, 0.0, lc};

Line(1) = {1, 2};  // bottom before step
Line(2) = {2, 3};  // step riser
Line(3) = {3, 4};  // bottom after step
Line(4) = {4, 5};  // outlet
Line(5) = {5, 6};  // top after step
Line(6) = {6, 7};  // step top riser
Line(7) = {7, 8};  // top before step
Line(8) = {8, 1};  // inlet

Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};

// Extrude in Z direction (0.01 m thickness for 2D)
Extrude {0, 0, 0.01} {
    Surface{1};
    Layers{1};
}

// After extrusion, Gmsh creates 10 surface entities (tags 1-10):
//   Surface 1  : front face  (z=0)   -> frontAndBack
//   Surface 2  : bottom before step  -> walls
//   Surface 3  : step riser           -> walls
//   Surface 4  : bottom after step   -> walls
//   Surface 5  : outlet face (x=6)   -> outlet
//   Surface 6  : top after step      -> walls
//   Surface 7  : step top riser      -> walls
//   Surface 8  : top before step     -> walls
//   Surface 9  : inlet face (x=0)    -> inlet
//   Surface 10 : back face (z=0.01)  -> frontAndBack

Physical Surface("frontAndBack") = {1, 10};
Physical Surface("inlet") = {9};
Physical Surface("outlet") = {5};
Physical Surface("walls") = {2, 3, 4, 6, 7, 8};
Physical Volume("FLUID") = {1};

Mesh 3;
