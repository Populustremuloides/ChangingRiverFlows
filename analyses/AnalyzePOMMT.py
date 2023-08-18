import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm


def analyzePOMMT():
    '''
    Calculate the changes in annual spectral properties of the
    discharge or temperature timeseries, per catchment, per year
    '''

    dataDict = {"catchment":[],"pommtSlope":[],"pommtMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the period of mean mangitude
        pommts = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            tempForWaterYear = np.array(ldf[tempVar])
            pommt = u_getPeriodOfMeanMagnitude(tempForWaterYear)
            pommts.append(pommt)

        slope = u_regressionFunction(waterYears, pommts)
        mean = np.mean(pommts)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["pommtSlope"].append(slope)
        dataDict["pommtMean"].append(mean)

        loop.set_description("Computing periods of mean mangitude for temperature")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_pommt.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
