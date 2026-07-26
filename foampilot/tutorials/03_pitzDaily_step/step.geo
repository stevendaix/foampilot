SetFactory("OpenCASCADE");

// Backward-facing step geometry (2D profile extruded)
// Reference: OpenFOAM pitzDaily

lc = 0.02;

// Step profile in XY plane (Z=0)
// Lower wall: (0,0) -> (1,0)
// Step vertical: (1,0) -> (1,0.5)
// Upper wall: (1,0.5) -> (6,0.5)
// Outlet wall: (6,0.5) -> (6,0.6)
// Upper outlet wall: (6,0.6) -> (1,0.6)
// Step top: (1,0.6) -> (1,1.0)
// Upper wall back: (1,1.0) -> (0,1.0)
// Inlet wall: (0,1.0) -> (0,0)

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {1.0, 0.0, 0.0, lc};
Point(3) = {1.0, 0.5, 0.0, lc};
Point(4) = {6.0, 0.5, 0.0, lc};
Point(5) = {6.0, 0.6, 0.0, lc};
Point(6) = {1.0, 0.6, 0.0, lc};
Point(7) = {1.0, 1.0, 0.0, lc};
Point(8) = {0.0, 1.0, 0.0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};

Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};

// Extrude to 3D
Extrude {0, 0, 0.01} {
  Surface{1};
  Layers{1};
}

Physical Surface("inlet") = {1};
Physical Surface("outlet") = {2};
Physical Surface("walls") = {3, 4, 5, 6};
Physical Volume("FLUID") = {1};

Mesh 3;
