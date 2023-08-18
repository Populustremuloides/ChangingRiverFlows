import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm


def analyzePOMMET():
    '''
    Calculate the changes in annual spectral properties of the
    evapotranspiration timeseries, per catchment, per year
    '''

    dataDict = {"catchment":[],"pommetSlope":[],"pommetMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the period of mean mangitude
        pommets = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            etForWaterYear = np.array(ldf[etVar])
            pommet = u_getPeriodOfMeanMagnitude(etForWaterYear)
            pommets.append(pommet)

        slope = u_regressionFunction(waterYears, pommets)
        mean = np.mean(pommets)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["pommetSlope"].append(slope)
        dataDict["pommetMean"].append(mean)

        loop.set_description("Computing periods of mean mangitude for et")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_pommet.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
