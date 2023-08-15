import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

from scipy.signal import blackman

#import matplotlib.pyplot as plt

def _getPeriodOfMeanMagnitude(flow):
    #flow = flow[:365] # ignore leap day because that will introduce bias?
    flow = flow / np.max(flow) # normalize
    flow = flow - np.mean(flow) # subtract the mean

    numDays = flow.shape[0]
    dt = 1. # one day per measurement
    periods = 1. / np.fft.rfftfreq(numDays, d=dt)[1:] # ignore the infinite time-horizon element

    w = blackman(flow.shape[0]) # use a blackman filter to reduce spectral leakage
    result = np.fft.rfft(flow * w)
    magnitudes = np.abs(result) ** 2

    magnitudes = magnitudes[1:] # ignore the infinite time horizon element

    periodOfMeanMagnitude = np.sum(periods * magnitudes) / periods.shape[0]

    return periodOfMeanMagnitude

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
