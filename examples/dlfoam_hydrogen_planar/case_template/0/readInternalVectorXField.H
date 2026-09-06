internalVectorXField #codeStream
{
    codeInclude
    #{
        #include "fvCFD.H"
        #include <vector>
        #include "flame_cpp_funcs.H"

    #};
    codeOptions
    #{
        -I$(LIB_SRC)/finiteVolume/lnInclude \
        -I$(LIB_SRC)/meshTools/lnInclude \
        -I$(FOAM_CASE)/0
    #};
    codeLibs
    #{
        -lmeshTools \
        -lfiniteVolume
    #};
    code
    #{
        const IOdictionary& d = static_cast<const IOdictionary&>(dict);
        const fvMesh& mesh = refCast<const fvMesh>(d.db());
        vectorField Y(mesh.nCells(), vector(0., 0., 0.));

        // Reading cantera data file
        std::vector<float> xc, Yc;
        read_2_column_data_file(dict.name()+".dat", xc, Yc);

        // initial perturbation: dr = A*sin(f*alpha)
        const float A_perturb = dict.lookupOrDefault<float>("amplitude", xc[xc.size()-1]/500); // [m] - amplitude of the perturbation
        const float f_perturb = dict.lookupOrDefault<float>("frequency", 0); // [-] - frequency of the perturbation
        const float Lx = dict.lookupOrDefault<float>("Lx", 0); // [m] - domain length
        const float Ly = dict.lookupOrDefault<float>("Ly", 1); // [m] - domain width
        const float x_front = dict.lookupOrDefault<float>("frontCoordinate", Lx/2); // [x] - Cantera data coordinate which will be mapped to center of the domain in OpenFOAM
        // Interpolating this data on the mesh
        forAll(Y, i)
        {
            const float y = mesh.C()[i][1];
            const float dx_perturb = A_perturb*sin(f_perturb*2*Foam::constant::mathematical::pi*y/Ly); // initial perturbation
            const float x = mesh.C()[i][0] - Lx/2 + x_front - dx_perturb;

            float Umag = interpolate(xc, Yc, x);
            Y[i] = vector(Umag,0,0);
            // Info << "data_x=" << x << ", OF_x=" << mesh.C()[i][0] << ": " << Y[i] << endl;
        }

        os << Y;
    #};

};
