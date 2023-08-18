import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm


def analyzePOMMF():
    '''
    Calculate the changes in annual spectral properties of the
    discharge or flow timeseries, per catchment, per year
    '''

    dataDict = {"catchment":[],"pommfSlope":[],"pommfMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the period of mean mangitude
        pommfs = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            flowForWaterYear = np.array(ldf[dischargeVar])
            pommf = u_getPeriodOfMeanMagnitude(flowForWaterYear)
            pommfs.append(pommf)

        slope = u_regressionFunction(waterYears, pommfs)
        mean = np.mean(pommfs)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["pommfSlope"].append(slope)
        dataDict["pommfMean"].append(mean)

        loop.set_description("Computing periods of mean mangitude for flow")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_pommf.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
