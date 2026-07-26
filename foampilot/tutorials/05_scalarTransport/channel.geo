SetFactory("OpenCASCADE");

// Channel flow geometry for scalar transport
// Reference: OpenFOAM scalarTransportFoam scalarTransport

// Channel dimensions
Lx = 20.0;
Ly = 1.0;
Lz = 0.01;

lc = 0.05;

Box(1) = {0.0, -0.5, -0.005, Lx, Ly, Lz};

Physical Surface("inlet") = {1};
Physical Surface("outlet") = {2};
Physical Surface("walls") = {3, 4};
Physical Surface("frontAndBack") = {5, 6};
Physical Volume("FLUID") = {1};

Mesh 3;
