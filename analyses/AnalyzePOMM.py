import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

from scipy.signal import blackman

#import matplotlib.pyplot as plt



def analyzePOMM():
    dataDict = {"catchment":[],"pommSlope":[],"pommMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)
        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the period of mean mangitude
        pomms = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            flowForWaterYear = np.array(ldf[dischargeVar])
            pomm = _getPeriodOfMeanMagnitude(flowForWaterYear)
            pomms.append(pomm)

        slope = u_regressionFunction(waterYears, pomms)
        mean = np.mean(pomms)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["pommSlope"].append(slope)
        dataDict["pommMean"].append(mean)

        loop.set_description("Computing periods of mean mangitude")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_pomm.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
