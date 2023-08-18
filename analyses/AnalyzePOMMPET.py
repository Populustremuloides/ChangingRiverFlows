import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm


def analyzePOMMPET():
    '''
    Calculate the changes in annual spectral properties of the
    potential evapotranspiration timeseries, per catchment, per year
    '''

    dataDict = {"catchment":[],"pommpetSlope":[],"pommpetMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the period of mean mangitude
        pommpets = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            petForWaterYear = np.array(ldf[petVar])
            pommpet = u_getPeriodOfMeanMagnitude(petForWaterYear)
            pommpets.append(pommpet)

        slope = u_regressionFunction(waterYears, pommpets)
        mean = np.mean(pommpets)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["pommpetSlope"].append(slope)
        dataDict["pommpetMean"].append(mean)

        loop.set_description("Computing periods of mean mangitude for pet")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_pommpet.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
