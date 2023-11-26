import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def calculateBudgetDeficits():
    '''
    Calculates the error between inputs and outputs: P - ET - Q 
    '''

    dataDict = {"catchment":[],"budget_deficit":[], "percent_deficit":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)

    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        df = df.groupby(waterYearVar).mean()

        etMean = np.mean(df[etVar])
        precipMean = np.mean(df[precipVar])
        dischargeMean = np.mean(df[dischargeVar])
        deficit = precipMean - etMean - dischargeMean
        percentDeficit = 100 * (deficit / dischargeMean)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["budget_deficit"].append(deficit)
        dataDict["percent_deficit"].append(percentDeficit)

        loop.set_description("Computing budget deficits")
        loop.update(1)

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_deficits.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
