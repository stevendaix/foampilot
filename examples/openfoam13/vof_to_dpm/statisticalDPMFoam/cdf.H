class CDF
{
public:
    DynamicList<DynamicList<scalar>> cdf;
    DynamicList<scalar> localMin, localMax;
    scalar globalMin, globalMax;
    label Div;
    label subDiv;

    CDF
        (
            const IOdictionary &myDict
        )
{
        //read globalMin/Max
        myDict.lookup("min_r") >> globalMin;
        myDict.lookup("max_r") >> globalMax;

        //read cdf
        myDict.lookup("Div") >> Div;
        myDict.lookup("subDiv") >> subDiv;

        for (auto i = 0; i < Div; ++i)
        {
            DynamicList<scalar> tmpCdf;
            for (auto j = 0; j < subDiv; ++j)
            {
                string a = "cdf" + std::to_string(i)+ "_" + std::to_string(j);
                scalar tmp = readScalar(myDict.lookup(a));
                tmpCdf.append(tmp);
            }
            cdf.append(tmpCdf);
        }
        //read localMin/Max
        for (auto i = 0; i < Div; ++i)
        {
            string a = "localMin" + std::to_string(i);
            string b = "localMax" + std::to_string(i);

            scalar lMin = readScalar(myDict.lookup(a));
            scalar lMax = readScalar(myDict.lookup(b));

            localMin.append(lMin);
            localMax.append(lMax);

        }
}
    ~CDF(){};
    
};

