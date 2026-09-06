inletValue #codeStream
{
    codeInclude
    #{
        #include "fvCFD.H"
    #};
    codeOptions
    #{
        -I$(LIB_SRC)/finiteVolume/lnInclude \
        -I$(LIB_SRC)/meshTools/lnInclude
    #};
    codeLibs
    #{
        -lmeshTools \
        -lfiniteVolume
    #};
    code
    #{
        scalar value;

        // Reading cantera data file
        ifstream dataFile(dict.name() + ".dat");
        if (! dataFile.is_open()){
            Info << "WARNING: Data file " + dict.name() + ".dat is not open!" << endl;
        } else {
            dataFile >> value; // skipping x
            Info << value;
            dataFile >> value;
            if (value < 0.) {
                Info << "WARNING: Negative value encountered in " + dict.name() + ".dat: " << value << endl;
                value = 0.;
            }
        }
        Info << "[" << dict.name() << "]: Setting " << value << " at inlet." << endl;
        os << value;
    #};

};
