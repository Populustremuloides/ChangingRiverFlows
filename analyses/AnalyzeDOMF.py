import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def _getDayOfMeanFlow(flow):
    dayInWaterYear = np.arange(flow.shape[0])
    dayOfMeanFlow = np.sum(dayInWaterYear * flow) / flow.shape[0]
    return dayOfMeanFlow

def analyzeDOMF():
    dataDict = {"catchment":[],"domfSlope":[],"domfMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the day of mean flow
        domfs = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            flowForWaterYear = np.array(ldf[dischargeVar])
            domf = _getDayOfMeanFlow(flowForWaterYear)
            domfs.append(domf)

        slope = u_regressionFunction(waterYears, domfs)
        mean = np.mean(domfs)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["domfSlope"].append(slope)
        dataDict["domfMean"].append(mean)

        loop.set_description("Computing days of mean flow")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "domfValues.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
