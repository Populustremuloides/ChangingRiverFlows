import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeDOPET():
    dataDict = {"catchment":[],"dopetSlope":[],"dopetMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the day of mean actual evapotranspiration
        dopets = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            etForWaterYear = np.array(ldf[etVar])
            dopet = (np.argmax(etForWaterYear) + 1) # +1 for 0 indexing in python
            dopets.append(dopet)

        slope = u_regressionFunction(waterYears, dopets)
        mean = np.mean(dopets)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["dopetSlope"].append(slope)
        dataDict["dopetMean"].append(mean)

        loop.set_description("Computing days of peak actual evapotranspiration")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_dopet.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
