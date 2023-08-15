import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeDOPF():
    dataDict = {"catchment":[],"dopfSlope":[],"dopfMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the day of mean flow
        dopfs = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            flowForWaterYear = np.array(ldf[dischargeVar])
            dopf = (np.argmax(flowForWaterYear) + 1) # +1 for 0 indexing in python
            dopfs.append(dopf)

        slope = u_regressionFunction(waterYears, dopfs)
        mean = np.mean(dopfs)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["dopfSlope"].append(slope)
        dataDict["dopfMean"].append(mean)

        loop.set_description("Computing days of peak flow")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_dopf.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
