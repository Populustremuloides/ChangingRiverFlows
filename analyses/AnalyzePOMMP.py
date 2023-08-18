import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm


def analyzePOMMP():
    '''
    Calculate the changes in annual spectral properties of the
    discharge or flow timeseries, per catchment, per year
    '''

    dataDict = {"catchment":[],"pommpSlope":[],"pommpMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the period of mean mangitude
        pommps = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            precipForWaterYear = np.array(ldf[precipVar])
            pommp = u_getPeriodOfMeanMagnitude(precipForWaterYear)
            pommps.append(pommp)

        slope = u_regressionFunction(waterYears, pommps)
        mean = np.mean(pommps)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["pommpSlope"].append(slope)
        dataDict["pommpMean"].append(mean)

        loop.set_description("Computing periods of mean mangitude for precip")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_pommp.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
