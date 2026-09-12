//Using standard c++ functions let's create the dictionary with all required variables  
//std::ofstream file("myOutputFile", std::ios::app); //for post process (information about dropplets is accumulated)
std::ofstream file("myChangingDict"); // for solver (every time new file)

//Head of the dict
file << "FoamFile" << std::endl << "{"<< std::endl;
file << '\t' << "version 2.0;" << std::endl;
file << '\t' << "format ascii;" << std::endl;
file << '\t' << "class dictionary;" << std::endl;
file << '\t' << "location \"\";" << std::endl; //change location if needed
file << '\t' << "object myChangingDict;" << std::endl; // use your name
file << "}" << std::endl;
// if dropplets are inside the specified region
if(ii >= 1)
{

// Number of dropplets
//file << "Time = " << runTime.timeName() << ";" << std::endl;
file << "N" << '\t' << i << ";" << std::endl;

//Position
printVectorList(dropPos, file, "position");
//Diameter
printScalarList(dropDiam, file, "d");
//Volume
printScalarList(dropVol, file, "V");
//Velocity
printVectorList(dropVel, file, "U");
}
else file << "N" << '\t' << 0 << ";" << std::endl;

file.close();
