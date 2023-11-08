import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeMASP():
    '''
    calculates the mean and slope values for
    masp: mean annual specific precipitation
    for each catchment, with years divided
    by a catchment-specific water year starting
    at the direst month of the year.
    '''

    dataDict = {"catchment":[],"maspSlope":[],"maspMean":[], "maspPercentChange":[]}


    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()
        masp = df[specificPrecipVar]
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, masp)
        mean = np.mean(masp)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["maspSlope"].append(slope)
        dataDict["maspMean"].append(mean)
        dataDict["maspPercentChange"].append(100 * (slope / mean))

        loop.set_description("Computing mean annual specific precipitation")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_masp.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
