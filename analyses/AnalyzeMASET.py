import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeMASET():
    '''
    calculates the mean and slope values for
    maset: mean annual specific evapotranspiration
    for each catchment, with years divided
    by a catchment-specific water year starting
    at the direst month of the year.
    '''

    dataDict = {"catchment":[],"masetSlope":[],"masetMean":[], "masetPercentChange":[]}


    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()
        maset = df[specificETVar]
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, maset)
        mean = np.mean(maset)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["masetSlope"].append(slope)
        dataDict["masetMean"].append(mean)
        dataDict["masetPercentChange"].append(100 * (slope / mean))

        loop.set_description("Computing mean annual specific evapotranspiration")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_maset.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
