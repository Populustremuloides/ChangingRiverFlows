import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeET():
    '''
    calculates the mean and slope values for
    the actual evapotranspiration
    for each catchment, with years divided
    by a catchment-specific water year starting
    at the direst month of the year.
    '''

    dataDict = {"catchment":[],"etSlope":[],"etMean":[], "etPercentChange":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()
        et = df[etVar]
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, et)
        mean = np.mean(et)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["etSlope"].append(slope)
        dataDict["etMean"].append(mean)
        dataDict["etPercentChange"].append(slope / mean)

        loop.set_description("Computing mean annual evapotranspiration changes")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_et.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
