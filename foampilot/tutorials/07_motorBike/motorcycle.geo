SetFactory("OpenCASCADE");

// Motorcycle aerodynamics - simplified geometry
// Reference: OpenFOAM motorBike

// Domain dimensions (wind tunnel)
Lx = 30.0;
Ly = 10.0;
Lz = 5.0;

lc = 0.5;

// Fluid domain
Box(1) = {-5.0, -5.0, 0.0, Lx, Ly, Lz};

// Simplified motorcycle body (box)
Box(2) = {2.0, -0.5, 0.3, 3.0, 1.0, 0.8};

// Subtract from fluid
BooleanDifference(200) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Physical Volume("FLUID") = {200};

Mesh 3;
