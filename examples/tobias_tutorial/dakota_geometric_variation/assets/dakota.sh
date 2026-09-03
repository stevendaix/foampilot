#!/bin/bash
#
# Description
#   Run the dakota tool and run the process chain
#
# ------------------------------------------------------------------------------
Tmean=320


# Dakota change the parameters in this file
# ------------------------------------------------------------------------------
dprepro $1 system/dakotaParameter.orig system/dakotaParameter


# Run simulation with new parameter set
# ------------------------------------------------------------------------------


    # Get angle1, angle2, length of inlet + loop number
    #---------------------------------------------------------------------------
    angle1=`head -5 system/dakotaParameter | tail -1 | cut -d'=' -f2`
    angle2=`head -6 system/dakotaParameter | tail -1 | cut -d'=' -f2`
    length=`head -7 system/dakotaParameter | tail -1 | cut -d'=' -f2`
    loopNumber=`cat .optimizationLoop`


    # Transform the scientific notation into a readable format for | bc
    # Here we remove e or E with *10^
    #---------------------------------------------------------------------------
    angle1=`echo $angle1 | sed -e 's/[eE]+*/\*10\^/'`
    angle2=`echo $angle2 | sed -e 's/[eE]+*/\*10\^/'`
    length=`echo $length | sed -e 's/[eE]+*/\*10\^/'`


    # Optical stuff
    #---------------------------------------------------------------------------
    angle1=`echo "scale=4; $angle1" | bc`
    angle2=`echo "scale=4; $angle2" | bc`
    length=`echo "scale=4; $length" | bc`



    # Output the parameters that are used now
    #---------------------------------------------------------------------------
    >&2 echo -e "   ++++ Evaluate sample $loopNumber"
    >&2 echo -e "   |"
    >&2 echo -e "   |--> baffle angle1 = $angle1 [degree]"
    >&2 echo -e "   |--> baffle angle2 = $angle2 [degree]"
    >&2 echo -e "   |--> length of cold inlet = $length [m]"


    # Set the new angle parameters for the baffles
    #---------------------------------------------------------------------------
    cp system/rotateBafflesDict system/rotateBaffles
    sed "s/angle1/$angle1/" system/rotateBaffles -i
    sed "s/angle2/$angle2/" system/rotateBaffles -i


    # Calculate the new coordinates for the inlet
    #---------------------------------------------------------------------------
    half=`echo "scale=6; $length/2" | bc`
    xCenter=0.035
    newX1=`echo "scale=6; $xCenter-$half" | bc`
    newX2=`echo "scale=6; $xCenter+$half" | bc`

    sed "22s/.*/x1 $newX1;/" system/blockMeshDict -i
    sed "23s/.*/x2 $newX2;/" system/blockMeshDict -i


    # Make loop folder for log files
    #---------------------------------------------------------------------------
    logFolder_="Log/Optimization"$loopNumber
    mkdir -p $logFolder_


    # Mesh the case with new parameters
    #---------------------------------------------------------------------------
    >&2 echo "   |--> Start new meshing"
    source createMesh > $logFolder_/meshing


    # Run the simulation
    #---------------------------------------------------------------------------
    >&2 echo "   |--> Start stimulation"
    foamRun > $logFolder_/solving


    # Get the minimum, maximum and average value of the temperature on the
    # outlet patch
    #---------------------------------------------------------------------------
    postProcess -latestTime -func patchOutletMin > Tmin
    postProcess -latestTime -func patchOutletMax > Tmax

    # Mass flow weighted mean temperature
    #---------------------------------------------------------------------------
    cat postProcessing/Taverage/0/surfaceFieldValue.dat | tail -1 | xargs | \
        cut -d' ' -f2 > Taverage

    Tmax=`cat Tmin | grep  "T = " | cut -d'=' -f2 | head -1`
    Tmin=`cat Tmax | grep  "T = " | cut -d'=' -f2 | head -1`
    Taverage=`cat Taverage`

    # Replace e in with *10^
    #---------------------------------------------------------------------------
    Taverage=`echo $Taverage | sed -e 's/[eE]+*/\*10\^/' | bc`


    # Prepare results (this may not work in all environments)
    # What I do is simply to copy the mesh and the results into a new time
    # folder that we can check out the simple calculations
    #--------------------------------------------------------------------------
    mkdir results/$loopNumber
    cp -r constant/polyMesh results/$loopNumber
    solutionFile=`ls -s | head -4 | tail -1 | xargs | cut -d' ' -f2`
    cp -r $solutionFile/* results/$loopNumber


    # Remove time directorys (reg expression would be nicer) and mesh
    #---------------------------------------------------------------------------
    rm -rf 0 1* 2* 3* 4* 5* 6* 7* 8* 9* constant/polyMesh #postProcess*


    # Resonse function:
    #   1:  the mean temperature should be achieved
    #   2:  the temperature distribution should be as good as possible
    # Both could be minimized (not done here)
    #---------------------------------------------------------------------------
    funct1=`echo "scale=6; $Tmean-$Taverage" | bc`
    funct2=`echo "scale=6; $Tmax-$Tmin" | bc`

    if [ `echo $funct1 | grep "-"` ];
    then
        funct1=${funct1//-}
    fi

    if [ `echo $funct2 | grep "-"` ];
    then
        funct2=${funct2//-}
    fi


    >&2 echo "   |--> Minimum Temperature at outlet is: $Tmin"
    >&2 echo "   |--> Maximum Temperature at outlet is: $Tmax"
    >&2 echo "   |--> Average Temperature at outlet is: $Taverage"
    >&2 echo "   |--> Angle of baffle is: $angle"
    >&2 echo "   |--> Function for dakota (Average) is: $funct1"
    >&2 echo "   |--> Function for dakota (Distribution) is: $funct2"
    >&2 echo "   |"
    #echo -e "$angle1\t$angle2\t$length\t$funct1\t$funct2" >> analyseData.dat
    echo -e "$funct1\t$funct2" > .dakotaInput.dak


    # Increase the loop number and store in dummy file
    #--------------------------------------------------------------------------
    echo $((loopNumber+1)) > .optimizationLoop


# Generate ouput file for DAKOTA's algorithm (Object function)
#------------------------------------------------------------------------------
cp .dakotaInput.dak $2

sleep 2.1


#------------------------------------------------------------------------------
