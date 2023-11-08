import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeD_P():
    '''
    Analyzes the ratio between discharge (d) and
    precipitation (p), sometimes called runoff ratio
    or water yield.
    '''

    dataDict = {"catchment":[],"d_pSlope":[],"d_pMean":[], "d_pPercentChange":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()

        runoffRatios = df[dischargeVar] / df[precipVar] # d (discharge) / p (precip)
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, runoffRatios)
        mean = np.mean(runoffRatios)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["d_pSlope"].append(slope)
        dataDict["d_pMean"].append(mean)
        dataDict["d_pPercentChange"].append(100 * (slope / mean))

        loop.set_description("Computing discharge / precip ratios")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_d_p.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
