import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzePET_P():
    '''
    Calculates the ratio of the mean annual potential evapotranspiration
    to the mean annual precipitation per catchment.
    '''

    dataDict = {"catchment":[],"pet_pSlope":[],"pet_pMean":[], "pet_pPercentChange":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()

        pet_ps = df[petVar] / df[precipVar]
        waterYears = list(df.index)

        slope = u_regressionFunction(waterYears, pet_ps)
        mean = np.mean(pet_ps)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["pet_pSlope"].append(slope)
        dataDict["pet_pMean"].append(mean)
        dataDict["pet_pPercentChange"].append(100 * (slope / mean))

        loop.set_description("Computing PET / P ratios")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_pet_p.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
