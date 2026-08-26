#ifndef floatingTurbinesFoam13_cellSetOption_H
#define floatingTurbinesFoam13_cellSetOption_H
#include "fvModel.H"
#include "fvCellZone.H"
#include "fvMatrix.H"
#include "volFields.H"
#include "boolList.H"
namespace Foam { namespace fv {
class cellSetOption : public fvModel
{
protected:
    const word name_;
    const word modelTypeLocal_;
    const fvMesh& mesh_;
    const Time& time_;
    dictionary dict_;
    dictionary coeffs_;
    wordList fieldNames_;
    boolList applied_;
    fvCellZone zone_;
    bool active_;
    virtual void addSup(fvMatrix<vector>&, const label) {}
    virtual void addSup(const volScalarField&, fvMatrix<vector>&, const label) {}
    virtual void addSup(fvMatrix<scalar>&, const label) {}
    virtual void addSup(const volScalarField&, fvMatrix<scalar>&, const label) {}
public:
    TypeName("cellSetOption");
    cellSetOption(const word& name, const word& modelType, const dictionary& dict, const fvMesh& mesh)
    : fvModel(name, modelType, mesh, dict), name_(name), modelTypeLocal_(modelType), mesh_(mesh), time_(mesh.time()),
      dict_(dict), coeffs_(fvModel::coeffs(modelType, dict)), fieldNames_(), applied_(), zone_(mesh), active_(true) { read(dict); }
    virtual ~cellSetOption() = default;
    virtual wordList addSupFields() const override { return fieldNames_; }
    virtual void addSup(const volScalarField& U, fvMatrix<scalar>& eqn) const override { const_cast<cellSetOption&>(*this).addSup(eqn, 0); }
    virtual void addSup(const volVectorField& U, fvMatrix<vector>& eqn) const override { const_cast<cellSetOption&>(*this).addSup(eqn, 0); }
    virtual void addSup(const volScalarField& rho, const volScalarField& U, fvMatrix<scalar>& eqn) const override { const_cast<cellSetOption&>(*this).addSup(rho, eqn, 0); }
    virtual void addSup(const volScalarField& rho, const volVectorField& U, fvMatrix<vector>& eqn) const override { const_cast<cellSetOption&>(*this).addSup(rho, eqn, 0); }
    virtual bool read(const dictionary& dict) override { dict_ = dict; coeffs_ = fvModel::coeffs(modelTypeLocal_, dict); active_ = dict.lookupOrDefault("active", true); return fvModel::read(dict); }
    virtual bool movePoints() override { zone_.movePoints(); return true; }
    virtual void topoChange(const polyTopoChangeMap& map) override { zone_.topoChange(map); }
    virtual void mapMesh(const polyMeshMap& map) override { zone_.mapMesh(map); }
    virtual void distribute(const polyDistributionMap& map) override { zone_.distribute(map); }
};
}}
#endif
