//number of dropplets rep timestep
//scalar d_mean = readScalar(myDict.lookup("d_mean"));
const scalar pos_x = readScalar(myDict.lookup("pos_x"));
scalar massFR = readScalar(myDict.lookup("massFR"));
scalar massFRcoeff = readScalar(myDict.lookup("massFRcoeff"));
scalar particle_rho = readScalar(kinematicCloud.particleProperties().subDict("constantProperties").lookup("rho0"));
//scalar vol_mean = d_mean*d_mean*d_mean*1.05;
//double tmp_drop_num = (massFR*runTime.deltaTValue())/(vol_mean*particle_rho);

//Info << "mean V - " << vol_mean << endl;
//Info << "num dr ex - " << tmp_drop_num << endl;

//Second random generator



//label drop_num = (int)tmp_drop_num; // the value should be calculated based on massFlowRate

//List of dropplets positions
DynamicList<vector> dropPos;
//List of dropplets diameter
DynamicList<scalar> dropDiam;
//List of dropplets Volume
DynamicList<scalar> dropVol;
//List of dropplets velosity
DynamicList<vector> dropVel;

//variable for probability variables
DynamicList<scalar> diamCDF;
scalar D_max;
scalar D_min;

CDF dCDF(dCDFDict);
CDF velXCDF(velXCDFDict);
CDF velYCDF(velRadCDFDict);
CDF velZCDF(velAzCDFDict);

//readFiles
D_max = readScalar(diamDistrDict.lookup("D_max"));
D_min = readScalar(diamDistrDict.lookup("D_min"));
//scalar tmpDiv = readScalar(diamDistrDict.lookup("Div"));
label Div;
diamDistrDict.lookup("Div") >> Div;
for (auto i = 0; i < Div; ++i)
{
    string a = "Distr" + std::to_string(i);
    scalar tmp = readScalar(diamDistrDict.lookup(a));
    diamCDF.append(tmp);
}


//introduction of dropplets
int ii = 0;
int i = 0;
scalar currentMFR = 0;
bool mfrLimit = true;
while (mfrLimit == true)
//for (label i = 0; i < drop_num; ++i)
{
    //vector pos;
    ++i;
    Info << "i - " << i << endl<< endl;
    scalar d = calcD(diamCDF, D_max, D_min, Div, randGen); 
    Info << "d - " << d << endl;
    scalar vol = 0.5236*d*d*d;
    currentMFR += vol*particle_rho;
    Info << "currentMFR - " << currentMFR << endl;
    if(currentMFR/runTime.deltaTValue() > (massFR*massFRcoeff)) mfrLimit = false;
    scalar r = calcR(dCDF, d, randGen);
    Info << "r - " << r << endl;


    //scalar pos_y = calcPosY(r, randGen); 
    //Info << "pos_y - " << pos_y << endl;
    //scalar pos_z = calcPosZ(r, pos_y, randGen);
    //Info << "pos_z - " << pos_z << endl;
    vector pos = position(pos_x,r,randGen);
    Info << "pos - " << pos << endl;
    scalar pos_y = pos.component(vector::Y);
    scalar pos_z = pos.component(vector::Z);
    //if(true){vector tmp(pos_x, pos_y, pos_z);pos = tmp;} 
    //vector pos(pos_x, pos_z, pos_y);
    vector vel = calcVel(velXCDF, velYCDF, velZCDF, r, pos_y, pos_z, randGen, d);
    Info << "vel - " << vel << endl;

    //#include "writeDrop.H" //use for data analisys
    dropPos.append(pos);
    dropVel.append(vel);
    dropVol.append(vol);
    dropDiam.append(d);
    ++ii;
}



#include "write.H" //use for solver



