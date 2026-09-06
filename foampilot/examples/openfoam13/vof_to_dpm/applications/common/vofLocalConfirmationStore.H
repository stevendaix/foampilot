#ifndef vofLocalConfirmationStore_H
#define vofLocalConfirmationStore_H

#include "regIOobject.H"
#include "DynamicList.H"
#include <cstdint>

namespace Foam
{

struct vofParcelConfirmation
{
    word cloudName;
    word alphaFieldName;
    std::uint64_t fragmentId;
    label ownerProc;
    label parcelsAdded;
    scalar massAdded;
    scalar expectedMass;
    scalarList speciesMassAdded;
    scalarList expectedSpeciesMass;
    bool success;

    vofParcelConfirmation()
    :
        cloudName(),
        alphaFieldName(),
        fragmentId(0),
        ownerProc(-1),
        parcelsAdded(0),
        massAdded(0),
        expectedMass(0),
        speciesMassAdded(),
        expectedSpeciesMass(),
        success(false)
    {}
};

inline bool operator!=(
    const vofParcelConfirmation& a,
    const vofParcelConfirmation& b
)
{
    return a.cloudName != b.cloudName
        || a.alphaFieldName != b.alphaFieldName
        || a.fragmentId != b.fragmentId
        || a.ownerProc != b.ownerProc
        || a.parcelsAdded != b.parcelsAdded
        || a.massAdded != b.massAdded
        || a.expectedMass != b.expectedMass
        || a.speciesMassAdded != b.speciesMassAdded
        || a.expectedSpeciesMass != b.expectedSpeciesMass
        || a.success != b.success;
}

inline Ostream& operator<<(Ostream& os, const vofParcelConfirmation& c)
{
    return os
        << c.cloudName
        << c.alphaFieldName
        << c.fragmentId
        << c.ownerProc
        << c.parcelsAdded
        << c.massAdded
        << c.expectedMass
        << c.speciesMassAdded
        << c.expectedSpeciesMass
        << c.success;
}

inline Istream& operator>>(Istream& is, vofParcelConfirmation& c)
{
    return is
        >> c.cloudName
        >> c.alphaFieldName
        >> c.fragmentId
        >> c.ownerProc
        >> c.parcelsAdded
        >> c.massAdded
        >> c.expectedMass
        >> c.speciesMassAdded
        >> c.expectedSpeciesMass
        >> c.success;
}

class vofLocalConfirmationStore
:
    public regIOobject
{
    DynamicList<vofParcelConfirmation> confirmations_;
    label timeIndex_;

public:
    TypeName("vofLocalConfirmationStore");

    vofLocalConfirmationStore
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
        confirmations_(),
        timeIndex_(-1)
    {}

    void begin(const label timeIndex)
    {
        if (timeIndex_ != timeIndex)
        {
            confirmations_.clear();
            timeIndex_ = timeIndex;
        }
    }

    void append(const vofParcelConfirmation& c)
    {
        confirmations_.append(c);
    }

    void clear(const label timeIndex)
    {
        confirmations_.clear();
        timeIndex_ = timeIndex;
    }

    const DynamicList<vofParcelConfirmation>& confirmations() const
    {
        return confirmations_;
    }

    label timeIndex() const
    {
        return timeIndex_;
    }

    virtual bool readData(Istream&)
    {
        return false;
    }

    virtual bool writeData(Ostream&) const
    {
        return true;
    }
};

} // End namespace Foam

#endif

// ************************************************************************* //
