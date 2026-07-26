SetFactory("OpenCASCADE");

// Building aerodynamics - simplified city quarter
// Reference: OpenFOAM buildingAirFlow

// Domain dimensions
Lx = 600.0;
Ly = 300.0;
Lz = 150.0;

lc = 10.0;

// Fluid domain
Box(1) = {0.0, 0.0, 0.0, Lx, Ly, Lz};

// Building 1
Box(2) = {100.0, 100.0, 0.0, 50.0, 50.0, 30.0};

// Building 2
Box(3) = {200.0, 100.0, 0.0, 50.0, 50.0, 40.0};

// Building 3
Box(4) = {300.0, 150.0, 0.0, 60.0, 60.0, 25.0};

// Building 4
Box(5) = {400.0, 100.0, 0.0, 40.0, 40.0, 35.0};

// Subtract buildings from fluid domain
BooleanDifference(6) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
BooleanDifference(7) = { Volume{6}; Delete; }{ Volume{3}; Delete; };
BooleanDifference(8) = { Volume{7}; Delete; }{ Volume{4}; Delete; };
BooleanDifference(9) = { Volume{8}; Delete; }{ Volume{5}; Delete; };

Physical Volume("FLUID") = {9};

Mesh 3;
