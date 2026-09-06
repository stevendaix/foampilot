from pathlib import Path

root = Path('/opt/openfoam13/src/lagrangian/parcel')

p = root/'parcelCloud/parcelCloud.H'
s = p.read_text()
s = s.replace('''            struct directParcelData
            {
                vector position;''', '''            struct directParcelData
            {
                word cloudName;
                label fragmentId;
                scalarList speciesMassFractions;
                vector position;''')
s = s.replace('''                directParcelData()
                :
                    position(Zero),''', '''                directParcelData()
                :
                    cloudName(),
                    fragmentId(-1),
                    speciesMassFractions(),
                    position(Zero),''')
p.write_text(s)

p = root/'parcelCloud/parcelCloudList.H'
s = p.read_text().replace('''    // Private data

        //- Reference to the mesh
        const fvMesh& mesh_;''', '''    // Private data

        //- Names corresponding to the parcel-cloud list entries
        const wordList cloudNames_;

        //- Reference to the mesh
        const fvMesh& mesh_;''')
p.write_text(s)

p = root/'parcelCloud/parcelCloudList.C'
s = p.read_text()
s = s.replace('''    PtrList<parcelCloud>(),
    mesh_(rho.mesh())''', '''    PtrList<parcelCloud>(),
    cloudNames_(cloudNames),
    mesh_(rho.mesh())''')
s = s.replace('''    if (size() == 1 && cloudName == defaultCloudName)
    {
        return operator[](0).commitDirect(data, injectorIndex);
    }

    FatalErrorInFunction
        << "Direct commit requires the default single parcel cloud; "
        << "requested " << cloudName << " with " << size()
        << " clouds" << exit(FatalError);

    return false;''', '''    forAll(cloudNames_, cloudI)
    {
        if (cloudNames_[cloudI] == cloudName)
        {
            return operator[](cloudI).commitDirect(data, injectorIndex);
        }
    }

    FatalErrorInFunction
        << "Direct commit requested unknown cloud " << cloudName
        << "; available clouds: " << cloudNames_ << exit(FatalError);

    return false;''')
p.write_text(s)

p = root/'parcelCloud/ParcelCloud.H'
s = p.read_text()
needle = '''            template<class P>
            static void setCp(P&, const scalar, long)
            {}

public:'''
replacement = '''            template<class P>
            static void setCp(P&, const scalar, long)
            {}

            template<class P>
            static auto setSpecies
            (P& parcel, const scalarList& values, int)
                -> decltype(parcel.Y() = values, void())
            {
                if (values.size() == parcel.Y().size())
                {
                    parcel.Y() = values;
                }
            }

            template<class P>
            static void setSpecies(P&, const scalarList&, long)
            {}

public:'''
if needle not in s:
    raise SystemExit('setCp anchor not found')
s = s.replace(needle, replacement)
s = s.replace('''                setCp(parcel, data.Cp, 0);
                checkProperties''', '''                setCp(parcel, data.Cp, 0);
                setSpecies(parcel, data.speciesMassFractions, 0);
                checkProperties''')
p.write_text(s)
print('extended multi-cloud API')
