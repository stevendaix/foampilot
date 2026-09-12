Mesh.MshFileVersion = 2.2;
Mesh.Algorithm3D = 10;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Merge "hull.stl";
ClassifySurfaces{40*Pi/180, 1, 1, Pi};
CreateTopology;
CreateGeometry;

// The imported DTC surface is expected to be closed after topology creation.
Surface Loop(1) = {Surface{:}};
Volume(1) = {1};

Field[1] = Distance;
Field[1].SurfacesList = {1};
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.015;
Field[2].SizeMax = 0.12;
Field[2].DistMin = 0.05;
Field[2].DistMax = 0.40;
Background Field = 2;

Mesh.CharacteristicLengthMin = 0.015;
Mesh.CharacteristicLengthMax = 0.12;
Mesh 3;
