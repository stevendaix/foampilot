#ifndef marineOversetProbe_H
#define marineOversetProbe_H

#include "fvMeshMover.H"
#include "autoPtr.H"
#include "MarineOversetInterpolation.H"
#include "MarineOversetMatrix.H"

namespace Foam
{
class MarineOversetCellState;

namespace fvMeshMovers
{

class marineOversetProbe
:
    public fvMeshMover
{
    autoPtr<MarineOversetCellState> state_;
    autoPtr<MarineOversetInterpolation> interpolation_;
    autoPtr<MarineOversetMatrix> matrixOperator_;

public:
    TypeName("marineOversetProbe");

    marineOversetProbe(fvMesh& mesh);
    marineOversetProbe(fvMesh& mesh, const dictionary& dict);
    marineOversetProbe(const marineOversetProbe&) = delete;
    virtual ~marineOversetProbe();

    virtual bool dynamic() const { return false; }
    virtual bool solidBody() const { return true; }
    virtual bool update();

    //- Apply strong overset constraints to a scalar equation
    void applyScalar(fvMatrix<scalar>& matrix) const;

    //- Apply strong overset constraints to a vector equation
    void applyVector(fvMatrix<vector>& matrix) const;
    virtual void topoChange(const polyTopoChangeMap&) {}
    virtual void mapMesh(const polyMeshMap&) {}
    virtual void distribute(const polyDistributionMap&) {}
    void operator=(const marineOversetProbe&) = delete;
};

}
}

#endif
