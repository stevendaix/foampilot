SetFactory("OpenCASCADE");

// Flow past cylinder in a channel
// Simplified external aerodynamics geometry

// Channel dimensions
Lx = 10.0;
Ly = 2.0;
Lz = 0.1;

lc = 0.05;

// Channel
Box(1) = {0.0, -1.0, -0.05, Lx, Ly, Lz};

// Cylindrical obstacle
Cylinder(2) = {3.0, 0.0, 0.0, 3.0, 0.0, 0.1, 0.25, 2*Pi};

// Boolean difference
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Physical Volume("FLUID") = {3};

Mesh 3;
