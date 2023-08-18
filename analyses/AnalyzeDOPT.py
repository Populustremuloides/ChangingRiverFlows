import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeDOPT():
    dataDict = {"catchment":[],"doptSlope":[],"doptMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the day of mean flow
        dopts = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            tempForWaterYear = np.array(ldf[tempVar])
            dopt = (np.argmax(tempForWaterYear) + 1) # +1 for 0 indexing in python
            dopts.append(dopt)

        slope = u_regressionFunction(waterYears, dopts)
        mean = np.mean(dopts)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["doptSlope"].append(slope)
        dataDict["doptMean"].append(mean)

        loop.set_description("Computing days of peak temperature")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_dopt.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
