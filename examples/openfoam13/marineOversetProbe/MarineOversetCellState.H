#ifndef MarineOversetCellState_H
#define MarineOversetCellState_H

#include "fvMesh.H"
#include "volFields.H"
#include "labelList.H"

namespace Foam
{

class MarineOversetCellState
{
public:
    enum cellStatus
    {
        calculated = 0,
        interpolated = 1,
        hole = 2
    };

private:
    const fvMesh& mesh_;
    labelList zoneId_;
    labelList status_;
    List<labelList> donorIndices_;
    List<scalarList> weights_;
    labelList acceptorIndices_;
    label nStencils_;

    void readAndValidate();
    void readAndValidateStencils();

public:
    explicit MarineOversetCellState(const fvMesh& mesh);

    const labelList& zoneId() const { return zoneId_; }
    const labelList& status() const { return status_; }
    label nZones() const;
    label nCalculated() const;
    label nInterpolated() const;
    label nHoles() const;
    label nStencils() const { return nStencils_; }
    const List<labelList>& donorIndices() const { return donorIndices_; }
    const List<scalarList>& weights() const { return weights_; }
    const labelList& acceptorIndices() const { return acceptorIndices_; }
};

}

#endif
