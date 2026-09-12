from pathlib import Path

p = Path('/home/ubuntu/foampilot/foampilot/examples/openfoam13/vof_to_dpm/applications/common/vofFragmentTransitionManager.C')
s = p.read_text()
start = s.index('labelList vofFragmentTransitionManager::globalCellIds() const\n{')
end = s.index('\n\nvofGlobalFragment vofFragmentTransitionManager::makeGlobalFragment', start)
new = r'''labelList vofFragmentTransitionManager::globalCellIds() const
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
}'''
p.write_text(s[:start] + new + s[end:])
print('fixed globalCellIds broadcast')
