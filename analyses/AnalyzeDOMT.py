import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeDOMT():
    dataDict = {"catchment":[],"domtSlope":[],"domtMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the day of mean temperature
        domts = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            tempForWaterYear = np.array(ldf[tempVar])
            domt = u_getDayOfMeanMagnitude(tempForWaterYear)
            domts.append(domt)

        slope = u_regressionFunction(waterYears, domts)
        mean = np.mean(domts)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["domtSlope"].append(slope)
        dataDict["domtMean"].append(mean)

        loop.set_description("Computing days of mean temperature")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_domt.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
