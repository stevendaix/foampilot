#ifndef MarineInterMeshStencilState_H
#define MarineInterMeshStencilState_H

#include "fvMesh.H"
#include "labelList.H"
#include "scalarList.H"
#include "word.H"

namespace Foam
{
class MarineInterMeshStencilState
{
    const fvMesh& mesh_;
    word donorRegion_;
    word acceptorRegion_;
    labelList acceptorIndices_;
    List<labelList> donorIndices_;
    List<scalarList> weights_;

public:
    MarineInterMeshStencilState
    (
        const fvMesh& mesh,
        const label donorCellCount
    );

    const word& donorRegion() const { return donorRegion_; }
    const word& acceptorRegion() const { return acceptorRegion_; }
    const labelList& acceptorIndices() const { return acceptorIndices_; }
    const List<labelList>& donorIndices() const { return donorIndices_; }
    const List<scalarList>& weights() const { return weights_; }
    label size() const { return acceptorIndices_.size(); }
};
}

#endif
