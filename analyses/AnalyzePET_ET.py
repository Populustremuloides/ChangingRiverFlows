import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzePET_ET():
    '''
    Calculates the ratio of the mean annual potential evapotranspiration
    to the mean annual actual evapotranspiration per catchment.
    '''

    dataDict = {"catchment":[],"pet_etSlope":[],"pet_etMean":[], "pet_etPercentChange":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()

        pet_ets = df[petVar] / df[etVar]
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, pet_ets)
        mean = np.mean(pet_ets)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["pet_etSlope"].append(slope)
        dataDict["pet_etMean"].append(mean)
        dataDict["pet_etPercentChange"].append(100 * (slope / mean))

        loop.set_description("Computing P / PET ratios")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_pet_et.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
