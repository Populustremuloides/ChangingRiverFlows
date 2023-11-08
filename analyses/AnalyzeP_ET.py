import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeP_ET():
    '''
    Calculates the ratio of the mean annual precipitation rate to the
    evapotranpsiration rate for each catchment.
    '''

    dataDict = {"catchment":[],"p_etSlope":[],"p_etMean":[], "p_etPercentChange":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()

        p_ets = df[precipVar] / df[etVar]
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, p_ets)
        mean = np.mean(p_ets)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["p_etSlope"].append(slope)
        dataDict["p_etMean"].append(mean)
        dataDict["p_etPercentChange"].append(100 * (slope / mean))

        loop.set_description("Computing P / ET ratios")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_p_et.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
