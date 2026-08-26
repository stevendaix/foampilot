/*---------------------------------------------------------------------------*\
  vofFragmentTransitionManager
  MPI-aware preparation of VOF fragments for direct parcel commit.
\*---------------------------------------------------------------------------*/
#ifndef vofFragmentTransitionManager_H
#define vofFragmentTransitionManager_H

#include "vofFragmentTransition.H"
#include "vofLocalConfirmationStore.H"
#include "globalIndex.H"
#include "Pstream.H"
#include "HashSet.H"
#include "DynamicList.H"
#include "SortableList.H"

namespace Foam
{

struct vofGlobalFragment
{
    uint64_t id;
    label ownerProc;
    DynamicList<label> globalCells;
    DynamicList<label> localCells;
    scalar volume;
    scalar mass;
    vector centroid;
    vector velocity;
    scalar enthalpy;
    scalar temperature;
    scalarList composition;

    vofGlobalFragment()
    :
        id(0), ownerProc(-1), globalCells(), localCells(), volume(0), mass(0),
        centroid(Zero), velocity(Zero), enthalpy(0), temperature(0),
        composition()
    {}
};

inline Ostream& operator<<(Ostream& os, const vofGlobalFragment& f)
{
    os  << f.id << f.ownerProc << f.globalCells << f.localCells
        << f.volume << f.mass
        << f.centroid << f.velocity << f.enthalpy << f.temperature
        << f.composition;
    return os;
}

inline Istream& operator>>(Istream& is, vofGlobalFragment& f)
{
    is  >> f.id >> f.ownerProc >> f.globalCells >> f.localCells
        >> f.volume >> f.mass
        >> f.centroid >> f.velocity >> f.enthalpy >> f.temperature
        >> f.composition;
    return is;
}

inline bool operator!=(const vofGlobalFragment& a, const vofGlobalFragment& b)
{
    return a.id != b.id;
}

struct vofBoundaryLink
{
    label procA;
    label procB;
    label fragmentA;
    label fragmentB;

    vofBoundaryLink()
    : procA(-1), procB(-1), fragmentA(0), fragmentB(0) {}
};

inline Ostream& operator<<(Ostream& os, const vofBoundaryLink& link)
{
    return os << link.procA << link.procB << link.fragmentA << link.fragmentB;
}

inline Istream& operator>>(Istream& is, vofBoundaryLink& link)
{
    return is >> link.procA >> link.procB >> link.fragmentA >> link.fragmentB;
}

inline bool operator!=(const vofBoundaryLink& a, const vofBoundaryLink& b)
{
    return a.fragmentA != b.fragmentA || a.fragmentB != b.fragmentB
        || a.procA != b.procA || a.procB != b.procB;
}

struct vofFragmentBatch
{
    label timeIndex;
    word cloudName;
    word alphaFieldName;
    word namespaceKey;
    List<vofGlobalFragment> fragments;

    vofFragmentBatch()
    :
        timeIndex(-1),
        cloudName(),
        alphaFieldName(),
        namespaceKey(),
        fragments()
    {}

    inline label size() const { return fragments.size(); }
};

class vofLocalTransitionBatch
:
    public regIOobject
{
    label timeIndex_;
    List<vofGlobalFragment> fragments_;

public:
    TypeName("vofLocalTransitionBatch");

    vofLocalTransitionBatch
    (
        const objectRegistry& db,
        const word& name
    )
    :
        regIOobject
        (
            IOobject
            (
                name,
                db.time().name(),
                db,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                true
            )
        ),
        timeIndex_(-1),
        fragments_()
    {}

    void reset(const vofFragmentBatch& batch, const label timeIndex)
    {
        timeIndex_ = timeIndex;
        fragments_.clear();
        forAll(batch.fragments, fragmentI)
        {
            const vofGlobalFragment& fragment = batch.fragments[fragmentI];
            if (fragment.ownerProc == Pstream::myProcNo())
            {
                fragments_.append(fragment);
            }
        }
    }

    label timeIndex() const { return timeIndex_; }
    const List<vofGlobalFragment>& fragments() const { return fragments_; }

    virtual bool readData(Istream&) { return false; }
    virtual bool writeData(Ostream&) const { return true; }
};

class vofFragmentTransitionManager
{
    const fvMesh& mesh_;
    const volScalarField& alpha_;
    const volVectorField& U_;
    const volScalarField& rho_;
    scalar threshold_;
    label minCells_;
    scalar minVolume_;
    scalar rhoLiquid_;
    word cloudName_;
    word alphaFieldName_;
    word namespaceKey_;
    scalarList speciesFractions_;
    mutable vofFragmentBatch lastBatch_;
    mutable label lastTimeIndex_;

    word objectName(const word& stem) const
    {
        return stem + "." + namespaceKey_;
    }

    labelList globalCellIds() const;
    vofGlobalFragment makeGlobalFragment
    (
        const vofFragmentTransitionRecord& local,
        const label ownerProc
    ) const;

public:
    vofFragmentTransitionManager
    (
        const fvMesh& mesh,
        const volScalarField& alpha,
        const volVectorField& U,
        const volScalarField& rho,
        const scalar threshold,
        const label minCells,
        const scalar minVolume,
        const scalar rhoLiquid,
        const word& cloudName = word(),
        const word& alphaFieldName = word(),
        const scalarList& speciesFractions = scalarList()
    );

    // Detect locally, reconcile processor-boundary fragments, and return a
    // deterministic global batch. No parcel is created by this method.
    // Exchange fragment labels on coupled processor faces. Each returned
    // link is an edge of the distributed fragment graph.
    List<vofBoundaryLink> boundaryLinksMPI
    (
        const List<vofFragmentTransitionRecord>& localFragments
    ) const;

    vofFragmentBatch reconcileMPI(const label timeIndex) const;

    const word& cloudName() const { return cloudName_; }
    const word& alphaFieldName() const { return alphaFieldName_; }
    const word& namespaceKey() const { return namespaceKey_; }
    const vofFragmentBatch& lastBatch() const { return lastBatch_; }

    // Reconcile local post-injection confirmations. This is collective only
    // when called by the fvModel, never from an InjectionModel callback.
    bool reconcileConfirmationsMPI
    (
        const vofFragmentBatch& batch,
        const List<vofParcelConfirmation>& localConfirmations,
        List<vofParcelConfirmation>& localResults
    ) const;

    // Deterministic owner rule used by commitToCloud(): smallest global cell.
    static label chooseOwner
    (
        const labelList& candidateProcs,
        const labelList& candidateCells
    );
};

} // End namespace Foam

#endif
