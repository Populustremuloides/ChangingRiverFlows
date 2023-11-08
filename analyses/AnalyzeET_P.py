import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeET_P():
    '''
    Calculates the ratio of the mean annual *actual* evapotranspiration
    to the mean annual actual evapotranspiration per catchment.
    '''

    dataDict = {"catchment":[],"et_pSlope":[],"et_pMean":[], "et_pPercentChange":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()

        et_ps = df[etVar] / df[precipVar]
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, et_ps)
        mean = np.mean(et_ps)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["et_pSlope"].append(slope)
        dataDict["et_pMean"].append(mean)
        dataDict["et_pPercentChange"].append(100 * (slope / mean))

        loop.set_description("Computing ET / P ratios")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_et_p.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
