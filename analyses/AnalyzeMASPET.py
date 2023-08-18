import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeMASPET():
    '''
    calculates the mean and slope values for
    maspet: mean annual potential evapotranspiration
    for each catchment, with years divided
    by a catchment-specific water year starting
    at the direst month of the year.
    '''

    dataDict = {"catchment":[],"maspetSlope":[],"maspetMean":[], "maspetPercentChange":[]}


    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()
        maspet = df[specificPETVar]
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, maspet)
        mean = np.mean(maspet)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["maspetSlope"].append(slope)
        dataDict["maspetMean"].append(mean)
        dataDict["maspetPercentChange"].append(slope / mean)

        loop.set_description("Computing mean annual specific potential evapotranspiration")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_maspet.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
