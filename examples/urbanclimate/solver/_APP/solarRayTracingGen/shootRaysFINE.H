// All rays expressed as start face (local) index end end face (global)
// Pre-size by assuming a certain percentage is visible.

// Maximum length for dynamicList
const label maxDynListLengthFINE
(
    viewFactorDict.lookupOrDefault<label>("maxDynListLength", 100000)
);

//for (label proci = 0; proci < Pstream::nProcs(); proci++)
//{
    // Shoot rays from me to proci. Note that even if processor has
    // 0 faces we still need to call findLine to keep calls synced.

    DynamicField<point> startFINE(mesh.nFaces());
    DynamicField<vector> startFINEArea(mesh.nFaces());
    DynamicField<point> endFINE(startFINE.size());
    DynamicList<label> startFINEIndex(startFINE.size());
    DynamicList<label> endFINEIndex(startFINE.size());

    const pointField& myFcFINE = localFINECf_[Pstream::myProcNo()];
    const vectorField& myAreaFINE = localFINESf_[Pstream::myProcNo()];

    //const pointField& remoteArea = remoteCoarseSf[proci];
    const pointField& remoteFcFINE = remoteFINECf_[Pstream::myProcNo()];

    label iFINE = 0;
    do
    {
        for (; iFINE < myFcFINE.size(); iFINE++)
        {
            const point& fc = myFcFINE[iFINE];
            const vector& area = myAreaFINE[iFINE];
            const point& remFc = remoteFcFINE[iFINE];
            
            const vector& d = remFc - fc;
            
            scalar surfTol = 0.001;
            scalar cellLen = Foam::sqrt(mag(area));
            if (surfTol*mag(d) < cellLen)
            {
                surfTol = max(0.001,(cellLen/(mag(d)+SMALL))/2); //to avoid start points very close to the surface
            }
            startFINE.append(fc + surfTol*d);
            startFINEIndex.append(iFINE);
            startFINEArea.append(area);
            endFINE.append(fc + 0.999*d);
            label globalI = globalNumbering.toGlobal(Pstream::myProcNo(), iFINE);
            endFINEIndex.append(globalI);
            if (startFINEIndex.size() > maxDynListLengthFINE)
            {
                break;
            }
        }
        
        List<pointIndexHit> hitInfo(startFINEIndex.size());
        surfacesMesh.findLine(startFINE, endFINE, hitInfo);
        
        forAll(hitInfo, rayI)
        {
            if (!hitInfo[rayI].hit()) // if the ray did not hit any surface between start and end positions...
            {
                scalar cosPhiTest = (startFINEArea[rayI] & sunPos)/(mag(startFINEArea[rayI])*mag(sunPos) + SMALL);
                if (cosPhiTest < 0) // ... and if the surface is looking towards the sun
                {
                    rayStartFaceFINE.append(startFINEIndex[rayI]);
                    rayEndFaceFINE.append(endFINEIndex[rayI]);
                    //Info << "Clear" << endl; //debug
                }
		        else
		        {
		            //Info << "Not looking towards the sun" << endl; //debug
		        }
            }
		    else
		    {
		        //Info << "Hit a wall" << endl; //debug
		    }
        }

        startFINE.clear();
        startFINEIndex.clear();
        endFINE.clear();
        endFINEIndex.clear();

    }while (returnReduce(iFINE < myFcFINE.size(), orOp<bool>()));

//}
