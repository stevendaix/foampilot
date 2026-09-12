void printVector
    (
        const vector& vec,
        std::ofstream& file,
        const string& varName
    )
{
    file << varName << '\t' << "(" << vec.component(vector::X) << " " << vec.component(vector::Y) << " " << vec.component(vector::Z) << ")" << ";" << std::endl << std::endl;
}

void printVector
    (
        const vector& vec,
        std::ofstream& file
    )
{
    file  << "(" << vec.component(vector::X) << " " << vec.component(vector::Y) << " " << vec.component(vector::Z) << ")";
}




void printVectorList
    (
        const DynamicList<vector>& vecList,
        std::ofstream& file,
        const string& varName
    )
{
    forAll(vecList,i)
    {
        file << varName << i << '\t';
        printVector(vecList[i], file);
        file << ";" <<std::endl;
    }
    file << std::endl;
}



void printScalarList
    (
        const DynamicList<scalar>& scalarList,
        std::ofstream& file,
        const string& varName
    )
{
    forAll(scalarList,i)
    {
        file << varName << i << '\t' << scalarList[i]<< ";" << std::endl;
    }
    file << std::endl;
}

scalar calcD
    (
        const DynamicList<scalar>& diamCDF,
        const scalar& D_max,
        const scalar& D_min,
        const int& Div,
        Foam::Random& ranGen
    )
{
    int rrr = ranGen.integer(1 , 99999);
    double rand1 = (double)rrr/10000.0;
    double rand = std::abs(Foam::cos(rand1));
    for(auto i = 0; i < Div; ++i)
    {
        if(diamCDF[i] > rand)
            {
                double a = (rand-diamCDF[i-1])/(diamCDF[i]-diamCDF[i-1]);
                return ((((double)(i)+a)/(double)Div)*(D_max-D_min));
            }
    }
    return 0;
}

scalar calcR
    (
        const CDF& dCDF,
        const scalar& d,
        Foam::Random& randGen
    )
{
    scalar a = d*(dCDF.Div-1);
    scalar b = dCDF.globalMax;

    scalar r1 = 0.0;
    scalar r2 = 0.0;
    scalar k = 0.0;    

    int i = (int)(a/b);
    double rand1 = (double)(randGen.integer(1 , 99999))/1000;
    double rand = std::abs(Foam::cos(rand1));
    for(auto j = 0; j < dCDF.subDiv; ++j)
    {
        if(dCDF.cdf[i][j] > rand)
            {
                r2 = (dCDF.localMin[i] + (((double)(j))/(double)dCDF.subDiv)*dCDF.localMax[i]);
                if(j == 0) r1 = dCDF.localMin[i];
                else r1 = (dCDF.localMin[i] + (((double)(j-1))/(double)dCDF.subDiv)*dCDF.localMax[i]);
                k = (rand - dCDF.cdf[i][j-1])/(dCDF.cdf[i][j] - dCDF.cdf[i][j-1]);

                return (k*r1+(1-k)*r2);
            }
    }
    return 0;
}


vector position
    (
        const scalar& pos_x,
        const scalar& r,
        Foam::Random& randGen 
    )
{
    double alpha = 50*M_PI*randGen.scalar01();
    scalar pos_y = r*Foam::cos(alpha);
    scalar pos_z = r*Foam::sin(alpha);
    vector res(pos_x, pos_y, pos_z);
    return res;
}

/*
scalar calcPosY
    (
        const scalar& r,
        Foam::Random& randGen
    )
{
    double angle = 2*M_PI*randGen.scalar01();
    double rand = (randGen.scalar01() - 0.5)*2;
    return rand*r;
}

scalar calcPosZ
    (
        const scalar& r,
        const scalar& y,
        Foam::Random& randGen
    )
{
    bool sign = randGen.integer(0 , 1);
    scalar tmp = (r*r)-(y*y);
    scalar z = Foam::sqrt(tmp);
    if (sign == 0) return -z;
    else return z;
}
*/
vector calcVel
    (
        const CDF& velXCDF,
        const CDF& velRadCDF,
        const CDF& velAzCDF,
        const scalar& r,
        const scalar& y,
        const scalar& z,
        Foam::Random& randGen,
        scalar& d
    )
{
    scalar vel_x = 0.0;
    scalar vel_Rad = 0.0;
    scalar vel_Az = 0.0;
    scalar alpha = 0.0;

    scalar v1 = 0.0;
    scalar v2 = 0.0;
    scalar k = 0.0;

    scalar a = (r - velXCDF.globalMin)*(velXCDF.Div-1); //-1
    scalar b = velXCDF.globalMax - velXCDF.globalMin;
    int i = (int)(a/b);
    double rand1 = (double)(randGen.integer(0 , 999999))/1000;
    double rand = std::abs(Foam::cos(rand1));

    for(auto j = 0; j < velXCDF.subDiv; ++j)
    {
        if(velXCDF.cdf[i][j] > rand && vel_x == 0.0)
        {
            v2 = (velXCDF.localMin[i] + ((double)j/(double)velXCDF.subDiv)*(velXCDF.localMax[i] - velXCDF.localMin[i]));
            if(j == 0) v1 = velXCDF.localMin[i];
            else v1 = (velXCDF.localMin[i] + ((double)(j-1)/(double)velXCDF.subDiv)*(velXCDF.localMax[i] - velXCDF.localMin[i]));
            k = (rand - velXCDF.cdf[i][j-1])/(velXCDF.cdf[i][j] - velXCDF.cdf[i][j-1]);
            vel_x = k*v1+(1-k)*v2;
        }
        if(velRadCDF.cdf[i][j] > rand && vel_Rad == 0.0) vel_Rad = (velRadCDF.localMin[i] + ((double)j/(double)velRadCDF.subDiv)*(velRadCDF.localMax[i] - velRadCDF.localMin[i]));
        if(velAzCDF.cdf[i][j] > rand && vel_Az == 0.0) vel_Az = (velAzCDF.localMin[i] + ((double)j/(double)velAzCDF.subDiv)*(velAzCDF.localMax[i] - velAzCDF.localMin[i]));
    }
    

    //transform Rad and Az velosity to Y and Z
    if(y > 0.0 && z > 0.0) alpha = Foam::atan((double)(std::abs(z)/std::abs(y)));
    if(y < 0.0 && z > 0.0) alpha = (M_PI*0.5 + Foam::atan((double)(std::abs(y)/std::abs(z))));
    if(y < 0.0 && z < 0.0) alpha = (M_PI + Foam::atan((double)(std::abs(z)/std::abs(y))));
    if(y > 0.0 && z < 0.0) alpha = (M_PI*1.5 +Foam::atan((double)(std::abs(y)/std::abs(z))));

    scalar vel_y = ((Foam::cos(alpha)*vel_Rad) + (-Foam::sin(alpha)*vel_Az)); //Rad Az
    scalar vel_z = ((Foam::cos(alpha)*vel_Az) + (Foam::sin(alpha)*vel_Rad)); //Az Rad

    vector vel(vel_x, vel_y, vel_z);
    return vel;
}







