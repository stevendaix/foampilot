SetFactory("OpenCASCADE");

// Thermal buoyancy in a room
// Reference: OpenFOAM buoyantSimpleFoam room

// Room dimensions
Lx = 5.0;
Ly = 4.0;
Lz = 3.0;

lc = 0.1;

// Room volume
Box(1) = {0.0, 0.0, 0.0, Lx, Ly, Lz};

// Physical groups
Physical Surface("inlet") = {1}; // Cold wall (x=0)
Physical Surface("hotWall") = {2}; // Hot wall (x=Lx)
Physical Surface("coldWall") = {3}; // Cold wall (y=0)
Physical Surface("walls") = {4, 5, 6}; // Top, bottom, other side
Physical Volume("FLUID") = {1};

Mesh 3;
