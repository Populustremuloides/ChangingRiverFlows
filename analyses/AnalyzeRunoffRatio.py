import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

# cycle through the data and calculate the runoff ratios
def analyzeRunoffRatio():
    dataDict = {"catchment":[],"runoffRatioSlope":[],"runoffRatioMean":[], "runoffRatioPercentChange":[]}


    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()

        runoffRatios = df[dischargeVar] / df[precipVar]
        waterYears = list(df.index) #[waterYearVar]

        slope = u_regressionFunction(waterYears, runoffRatios)
        mean = np.mean(runoffRatios)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["runoffRatioSlope"].append(slope)
        dataDict["runoffRatioMean"].append(mean)
        dataDict["runoffRatioPercentChange"].append(slope / mean)

        loop.set_description("Computing runoff ratios")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_runoffRatio.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
