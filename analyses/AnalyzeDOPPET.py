import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeDOPPET():
    dataDict = {"catchment":[],"doppetSlope":[],"doppetMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the day of mean actual evapotranspiration
        doppets = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            petForWaterYear = np.array(ldf[petVar])
            doppet = (np.argmax(petForWaterYear) + 1) # +1 for 0 indexing in python
            doppets.append(doppet)

        slope = u_regressionFunction(waterYears, doppets)
        mean = np.mean(doppets)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["doppetSlope"].append(slope)
        dataDict["doppetMean"].append(mean)

        loop.set_description("Computing days of peak potential evapotranspiration")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_doppet.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
