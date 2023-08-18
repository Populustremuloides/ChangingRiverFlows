import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeDOMET():
    dataDict = {"catchment":[],"dometSlope":[],"dometMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the day of mean evapotranspiration
        domets = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            etForWaterYear = np.array(ldf[etVar])
            domet = u_getDayOfMeanMagnitude(etForWaterYear)
            domets.append(domet)

        slope = u_regressionFunction(waterYears, domets)
        mean = np.mean(domets)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["dometSlope"].append(slope)
        dataDict["dometMean"].append(mean)

        loop.set_description("Computing days of mean evapotranspiration")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_domet.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
