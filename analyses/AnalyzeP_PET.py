import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeP_PET():
    '''
    Calculates the ratio of the mean annual precipitation rate to the
    potential evapotranpsiration rate for each catchment.
    '''

    dataDict = {"catchment":[],"p_petSlope":[],"p_petMean":[], "p_petPercentChange":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()

        p_pets = df[precipVar] / df[petVar]
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, p_pets)
        mean = np.mean(p_pets)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["p_petSlope"].append(slope)
        dataDict["p_petMean"].append(mean)
        dataDict["p_petPercentChange"].append(slope / mean)

        loop.set_description("Computing P / PET ratios")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_p_pet.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
