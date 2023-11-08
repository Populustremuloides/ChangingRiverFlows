import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

import matplotlib.pyplot as plt

def analyzeMASD():
    '''
    calculates the mean and slope values for
    masd: mean annual specific discharge
    for each catchment, with years divided
    by a catchment-specific water year starting
    at the direst month of the year.
    '''

    dataDict = {"catchment":[],"masdSlope":[],"masdMean":[], "masdPercentChange":[]}


    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()
        masd = df[specificDischargeVar]
        waterYears = list(df.index)
        
        slope = u_regressionFunction(waterYears, masd)
        mean = np.mean(masd)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["masdSlope"].append(slope)
        dataDict["masdMean"].append(mean)
        dataDict["masdPercentChange"].append(100 * (slope / mean))

        loop.set_description("Computing mean annual specific discharges")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_masd.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
