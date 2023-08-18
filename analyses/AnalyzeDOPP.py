import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeDOPP():
    dataDict = {"catchment":[],"doppSlope":[],"doppMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the day of mean flow
        dopps = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            precipForWaterYear = np.array(ldf[precipVar])
            dopp = (np.argmax(precipForWaterYear) + 1) # +1 for 0 indexing in python
            dopps.append(dopp)

        slope = u_regressionFunction(waterYears, dopps)
        mean = np.mean(dopps)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["doppSlope"].append(slope)
        dataDict["doppMean"].append(mean)

        loop.set_description("Computing days of peak precip")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_dopp.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
