#include "vofFragmentTransitionManager.H"
#include "vofLocalConfirmationStore.H"
#include "Pstream.H"
#include "syncTools.H"
#include "processorPolyPatch.H"
#include "ListOps.H"
#include "HashTable.H"
#include "mathematicalConstants.H"

namespace Foam
{

defineTypeNameAndDebug(vofLocalTransitionBatch, 0);
defineTypeNameAndDebug(vofLocalConfirmationStore, 0);

vofFragmentTransitionManager::vofFragmentTransitionManager
(
    const fvMesh& mesh,
    const volScalarField& alpha,
    const volVectorField& U,
    const volScalarField& rho,
    const scalar threshold,
    const label minCells,
    const scalar minVolume,
    const scalar rhoLiquid,
    const word& cloudName,
    const word& alphaFieldName,
    const scalarList& speciesFractions
)
:
    mesh_(mesh),
    alpha_(alpha),
    U_(U),
    rho_(rho),
    threshold_(threshold),
    minCells_(minCells),
    minVolume_(minVolume),
    rhoLiquid_(rhoLiquid),
    cloudName_(cloudName),
    alphaFieldName_(alphaFieldName),
    namespaceKey_(cloudName + "." + alphaFieldName),
    speciesFractions_(speciesFractions),
    lastBatch_(),
    lastTimeIndex_(-1)
{
    if (cloudName_.empty() || alphaFieldName_.empty())
    {
        FatalErrorInFunction
            << "vofFragmentTransitionManager requires non-empty cloudName "
            << "and alphaFieldName" << exit(FatalError);
    }
}


labelList vofFragmentTransitionManager::globalCellIds() const
{
    // Build a decomposition-independent numbering from cell centres.
    const vectorField& centres = mesh_.C();
    List<List<point>> gathered(Pstream::nProcs());
    gathered[Pstream::myProcNo()] = List<point>(centres);
    Pstream::gatherList(gathered);

    List<labelList> perProc(Pstream::nProcs());
    if (Pstream::master())
    {
        label total = 0;
        forAll(gathered, procI)
        {
            total += gathered[procI].size();
        }
        List<label> procOf(total, -1);
        List<label> localOf(total, -1);
        List<point> allCentres(total);
        label node = 0;
        forAll(gathered, procI)
        {
            forAll(gathered[procI], cellI)
            {
                procOf[node] = procI;
                localOf[node] = cellI;
                allCentres[node] = gathered[procI][cellI];
                ++node;
            }
        }
        labelList order(total);
        forAll(order, i)
        {
            order[i] = i;
        }
        Foam::sort(order, [&](const label a, const label b)
        {
            const point& pa = allCentres[a];
            const point& pb = allCentres[b];
            if (pa.x() != pb.x()) return pa.x() < pb.x();
            if (pa.y() != pb.y()) return pa.y() < pb.y();
            if (pa.z() != pb.z()) return pa.z() < pb.z();
            FatalErrorInFunction
                << "Coincident cell centres prevent a unique global cell numbering"
                << exit(FatalError);
            return a < b;
        });
        forAll(perProc, procI)
        {
            perProc[procI].setSize(gathered[procI].size());
        }
        forAll(order, globalCellI)
        {
            perProc[procOf[order[globalCellI]]][localOf[order[globalCellI]]]
                = globalCellI;
        }
    }

    // A nested List<labelList> is not reliably distributed by scatterList in
    // every OpenFOAM 13 communication mode. Broadcast one rank payload at a
    // time instead; all ranks execute the same collective sequence.
    labelList localIds;
    for (label targetProc = 0; targetProc < Pstream::nProcs(); ++targetProc)
    {
        labelList payload;
        if (Pstream::master())
        {
            payload = perProc[targetProc];
        }
        Pstream::scatter(payload);
        if (targetProc == Pstream::myProcNo())
        {
            localIds = payload;
        }
    }
    return localIds;
}

vofGlobalFragment vofFragmentTransitionManager::makeGlobalFragment
(
    const vofFragmentTransitionRecord& local,
    const label ownerProc
) const
{
    vofGlobalFragment result;
    result.id = local.id;
    result.ownerProc = ownerProc;
    result.globalCells = local.globalCells;
    if (ownerProc == Pstream::myProcNo())
    {
        result.localCells = local.cells;
    }
    result.volume = local.volume;
    result.mass =
        rhoLiquid_ > 0
      ? rhoLiquid_*local.volume
      : scalar(0);
    result.centroid = local.centroid;
    result.velocity = local.velocity;
    result.composition = speciesFractions_;
    return result;
}


label vofFragmentTransitionManager::chooseOwner
(
    const labelList& candidateProcs,
    const labelList& candidateCells
)
{
    if (candidateProcs.empty() || candidateCells.empty())
    {
        return -1;
    }

    label owner = candidateProcs[0];
    label smallestCell = candidateCells[0];

    forAll(candidateProcs, i)
    {
        if (candidateCells[i] < smallestCell)
        {
            smallestCell = candidateCells[i];
            owner = candidateProcs[i];
        }
        else if
        (
            candidateCells[i] == smallestCell
         && candidateProcs[i] < owner
        )
        {
            owner = candidateProcs[i];
        }
    }
    return owner;
}


List<vofBoundaryLink> vofFragmentTransitionManager::boundaryLinksMPI
(
    const List<vofFragmentTransitionRecord>& localFragments
) const
{
    DynamicList<vofBoundaryLink> links;
    labelList cellFragment(mesh_.nCells(), -1);
    // syncTools applies geometric transforms to exchanged values. Use a
    // label graph key here; the persistent uint64 fragment ID remains in the
    // gathered fragment records.
    labelList cellIds(mesh_.nCells(), -1);

    forAll(localFragments, fragmentI)
    {
        forAll(localFragments[fragmentI].cells, cellI)
        {
            const label celli = localFragments[fragmentI].cells[cellI];
            if (celli >= 0 && celli < mesh_.nCells())
            {
                cellFragment[celli] = fragmentI;
                cellIds[celli] = fragmentI;
            }
        }
    }

    labelList neighbourIds;
    syncTools::swapBoundaryCellList(mesh_, cellIds, neighbourIds);
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    const label myProc = Pstream::myProcNo();

    forAll(patches, patchI)
    {
        const polyPatch& patch = patches[patchI];
        if (!isA<processorPolyPatch>(patch))
        {
            continue;
        }
        const processorPolyPatch& procPatch = refCast<const processorPolyPatch>(patch);
        const label neighbourProc = procPatch.neighbProcNo();
        const labelUList& faceCells = patch.faceCells();
        forAll(faceCells, faceI)
        {
            const label celli = faceCells[faceI];
            const label boundaryFaceI =
                patch.start() + faceI - mesh_.nInternalFaces();
            const label localFragment = cellIds[celli];
            const label neighbourFragment = neighbourIds[boundaryFaceI];
            if (localFragment < 0 || neighbourFragment < 0)
            {
                continue;
            }

            vofBoundaryLink link;
            link.procA = myProc;
            link.procB = neighbourProc;
            // Store rank-local fragment indices in the edge. The union-find
            // stage resolves them to persistent fragment IDs after gathering
            // the rank-local record tables.
            link.fragmentA = localFragment;
            link.fragmentB = neighbourFragment;
            if
            (
                link.procB < link.procA
             || (link.procB == link.procA && link.fragmentB < link.fragmentA)
            )
            {
                Swap(link.procA, link.procB);
                Swap(link.fragmentA, link.fragmentB);
            }
            links.append(link);
        }
    }
    return links;
}


vofFragmentBatch vofFragmentTransitionManager::reconcileMPI
(
    const label timeIndex
) const
{
    if (lastTimeIndex_ == timeIndex)
    {
        return lastBatch_;
    }

    vofFragmentBatch batch;
    batch.timeIndex = timeIndex;
    batch.cloudName = cloudName_;
    batch.alphaFieldName = alphaFieldName_;
    batch.namespaceKey = namespaceKey_;

    // This is intentionally collective. Build the decomposition-independent
    // cell numbering before assigning fragment identities.
    const labelList globalIds = globalCellIds();
    const List<vofFragmentTransitionRecord> localFragments =
        vofFragmentTransition::detect
        (
            alpha_,
            U_,
            threshold_,
            minCells_,
            minVolume_,
            &globalIds
        );

    // All ranks must call reconcileMPI once per time index, in the same order.
    List<vofBoundaryLink> localLinks = boundaryLinksMPI(localFragments);
    List<List<vofFragmentTransitionRecord>> gathered(Pstream::nProcs());
    gathered[Pstream::myProcNo()] = localFragments;
    Pstream::gatherList(gathered);

    List<List<vofBoundaryLink>> gatheredLinks(Pstream::nProcs());
    gatheredLinks[Pstream::myProcNo()] = localLinks;
    Pstream::gatherList(gatheredLinks);

    List<vofGlobalFragment> globalFragments;
    if (Pstream::master())
    {
        labelList offsets(Pstream::nProcs() + 1, 0);
        forAll(gathered, procI)
        {
            offsets[procI + 1] = offsets[procI] + gathered[procI].size();
        }

        labelList parent(offsets.last());
        forAll(parent, nodeI)
        {
            parent[nodeI] = nodeI;
        }

        auto findRoot = [&](label node)
        {
            label root = node;
            while (parent[root] != root)
            {
                root = parent[root];
            }
            while (parent[node] != node)
            {
                const label next = parent[node];
                parent[node] = root;
                node = next;
            }
            return root;
        };

        forAll(gatheredLinks, procI)
        {
            forAll(gatheredLinks[procI], linkI)
            {
                const vofBoundaryLink& link = gatheredLinks[procI][linkI];
                if
                (
                    link.procA < 0 || link.procB < 0
                 || link.procA >= Pstream::nProcs()
                 || link.procB >= Pstream::nProcs()
                 || link.fragmentA < 0
                 || link.fragmentB < 0
                 || link.fragmentA >= gathered[link.procA].size()
                 || link.fragmentB >= gathered[link.procB].size()
                )
                {
                    continue;
                }
                const label nodeA = offsets[link.procA] + link.fragmentA;
                const label nodeB = offsets[link.procB] + link.fragmentB;
                const label rootA = findRoot(nodeA);
                const label rootB = findRoot(nodeB);
                if (rootA != rootB)
                {
                    parent[max(rootA, rootB)] = min(rootA, rootB);
                }
            }
        }

        HashTable<vofGlobalFragment, label> byComponent;
        forAll(gathered, procI)
        {
            const List<vofFragmentTransitionRecord>& records = gathered[procI];
            forAll(records, recordI)
            {
                const vofFragmentTransitionRecord& local = records[recordI];
                const label root = findRoot(offsets[procI] + recordI);
                if (!byComponent.found(root))
                {
                    byComponent.insert
                    (
                        root,
                        makeGlobalFragment(local, procI)
                    );
                    continue;
                }

                vofGlobalFragment& global = byComponent[root];
                const scalar oldVolume = global.volume;
                global.volume += local.volume;
                global.mass += rhoLiquid_ > 0 ? rhoLiquid_*local.volume : scalar(0);
                global.centroid =
                    (global.centroid*oldVolume
                   + local.centroid*local.volume)/max(global.volume, SMALL);
                global.velocity =
                    (global.velocity*oldVolume
                   + local.velocity*local.volume)/max(global.volume, SMALL);
                forAll(local.globalCells, cellI)
                {
                    global.globalCells.append(local.globalCells[cellI]);
                }
                if (procI == global.ownerProc)
                {
                    global.localCells = local.cells;
                }
                global.ownerProc = min(global.ownerProc, procI);
            }
        }

        // The component ID must be derived from the complete global cell set,
        // not from the first rank-local record encountered by the gather.
        // This keeps IDs invariant under MPI rank count and decomposition.
        typedef HashTable<vofGlobalFragment, label> fragmentTable;
        forAllConstIter(fragmentTable, byComponent, iter)
        {
            vofGlobalFragment& global = byComponent[iter.key()];
            global.id = std::numeric_limits<std::uint64_t>::max();
            forAll(global.globalCells, cellI)
            {
                global.id = min
                (
                    global.id,
                    static_cast<std::uint64_t>(global.globalCells[cellI])
                );
            }
        }

        forAllConstIter(fragmentTable, byComponent, iter)
        {
            globalFragments.append(iter());
        }
    }

    // Scatter the same deterministic global batch to every rank. A later
    // production implementation may scatter only owner-local records, but
    // keeping the full batch here makes ownership and auditing explicit.
    List<List<vofGlobalFragment>> perProc(Pstream::nProcs());
    if (Pstream::master())
    {
        forAll(perProc, procI)
        {
            perProc[procI] = globalFragments;
        }
    }
    Pstream::scatterList(perProc);
    batch.fragments = perProc[Pstream::myProcNo()];

    // Stable ordering makes parcel indices independent of HashTable order.
    SortableList<uint64_t> ids(batch.fragments.size());
    forAll(batch.fragments, i)
    {
        ids[i] = batch.fragments[i].id;
    }
    ids.sort();
    List<vofGlobalFragment> ordered(batch.fragments.size());
    forAll(ids, i)
    {
        forAll(batch.fragments, j)
        {
            if (batch.fragments[j].id == ids[i])
            {
                ordered[i] = batch.fragments[j];
                break;
            }
        }
    }
    batch.fragments.transfer(ordered);
    lastBatch_ = batch;
    lastTimeIndex_ = timeIndex;
    return lastBatch_;
}

bool vofFragmentTransitionManager::reconcileConfirmationsMPI
(
    const vofFragmentBatch& batch,
    const List<vofParcelConfirmation>& localConfirmations,
    List<vofParcelConfirmation>& localResults
) const
{
    const label nProcs = Pstream::nProcs();
    const label myProc = Pstream::myProcNo();

    List<List<vofParcelConfirmation>> gathered(nProcs);
    gathered[myProc] = localConfirmations;
    Pstream::gatherList(gathered);

    List<List<vofParcelConfirmation>> statuses(nProcs);
    label globalSuccess = 1;

    if (Pstream::master())
    {
        HashTable<vofParcelConfirmation, word> byTransaction;
        HashSet<word> duplicateTransactions;

        forAll(gathered, procI)
        {
            forAll(gathered[procI], confirmationI)
            {
                const vofParcelConfirmation& confirmation =
                    gathered[procI][confirmationI];
                const word key =
                    confirmation.cloudName + "."
                  + confirmation.alphaFieldName + "."
                  + Foam::name(confirmation.fragmentId);

                if (byTransaction.found(key))
                {
                    duplicateTransactions.insert(key);
                }
                else
                {
                    byTransaction.insert(key, confirmation);
                }
            }
        }

        forAll(batch.fragments, fragmentI)
        {
            const vofGlobalFragment& fragment = batch.fragments[fragmentI];
            vofParcelConfirmation status;
            status.cloudName = batch.cloudName;
            status.alphaFieldName = batch.alphaFieldName;
            status.fragmentId = fragment.id;
            status.ownerProc = fragment.ownerProc;
            status.expectedMass = fragment.mass;
            status.expectedSpeciesMass.setSize(fragment.composition.size(), 0);
            forAll(fragment.composition, speciesI)
            {
                status.expectedSpeciesMass[speciesI] =
                    fragment.mass*fragment.composition[speciesI];
            }
            status.success = false;

            const word key =
                batch.cloudName + "."
              + batch.alphaFieldName + "."
              + Foam::name(fragment.id);
            bool valid =
                fragment.ownerProc >= 0
             && fragment.ownerProc < nProcs
             && !duplicateTransactions.found(key)
             && byTransaction.found(key);

            if (valid)
            {
                const vofParcelConfirmation& confirmation =
                    byTransaction[key];
                const scalar tolerance =
                    1e-8*max(mag(fragment.mass), scalar(1));

                status = confirmation;
                valid =
                    confirmation.ownerProc == fragment.ownerProc
                 && confirmation.success
                    && confirmation.cloudName == batch.cloudName
                    && confirmation.alphaFieldName == batch.alphaFieldName
                    && confirmation.parcelsAdded == 1
                    && mag(confirmation.massAdded - fragment.mass)
                    <= tolerance
                 && mag(confirmation.massAdded - confirmation.expectedMass)
                    <= tolerance
                 && confirmation.speciesMassAdded.size()
                    == status.expectedSpeciesMass.size()
                 && confirmation.expectedSpeciesMass.size()
                    == status.expectedSpeciesMass.size();

                if (valid)
                {
                    forAll(status.expectedSpeciesMass, speciesI)
                    {
                        valid =
                            valid
                         && mag
                            (
                                confirmation.speciesMassAdded[speciesI]
                              - status.expectedSpeciesMass[speciesI]
                            ) <= tolerance;
                    }
                }
            }

            status.success = valid;
            if (fragment.ownerProc >= 0 && fragment.ownerProc < nProcs)
            {
                statuses[fragment.ownerProc].append(status);
            }
            globalSuccess = globalSuccess && valid;
        }
    }

    // Diffuse the single global decision as a scalar. scatterList() is
    // reserved for the per-owner status lists below.
    Pstream::scatter(globalSuccess);
    Pstream::scatterList(statuses);

    localResults = statuses[myProc];
    bool localSuccess = globalSuccess != 0;
    forAll(localResults, resultI)
    {
        localSuccess = localSuccess && localResults[resultI].success;
    }
    return localSuccess;
}


} // End namespace Foam
