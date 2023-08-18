import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeDOMP():
    dataDict = {"catchment":[],"dompSlope":[],"dompMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the day of mean precipitation
        domps = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            precipForWaterYear = np.array(ldf[precipVar])
            domp = u_getDayOfMeanMagnitude(precipForWaterYear)
            domps.append(domp)

        slope = u_regressionFunction(waterYears, domps)
        mean = np.mean(domps)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["dompSlope"].append(slope)
        dataDict["dompMean"].append(mean)

        loop.set_description("Computing days of mean precipitation")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_domp.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
