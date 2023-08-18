import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeMAT():
    '''
    calculates the mean and slope values for
    the mean annual temperature
    for each catchment, with years divided
    by a catchment-specific water year starting
    at the direst month of the year.
    '''

    dataDict = {"catchment":[],"matSlope":[],"matMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()
        mat = df[tempVar]
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, mat)
        mean = np.mean(mat)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["matSlope"].append(slope)
        dataDict["matMean"].append(mean)

        loop.set_description("Computing mean annual temperature changes")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_mat.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
