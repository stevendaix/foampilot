SetFactory("OpenCASCADE");

// Dam break geometry (rectangular tank)
// Reference: OpenFOAM interFoam damBreak

// Tank dimensions
Lx = 5.5;
Ly = 0.4;
Lz = 0.5;
Lz_water = 0.25; // Initial water height

lc = 0.05;

// Full tank
Box(1) = {0.0, 0.0, 0.0, Lx, Ly, Lz};

// Physical groups
Physical Surface("inlet") = {1};   // x=0
Physical Surface("outlet") = {2};  // x=Lx
Physical Surface("walls") = {3, 4, 5, 6}; // y=0, y=Ly, z=0, z=Lz
Physical Volume("FLUID") = {1};

Mesh 3;
