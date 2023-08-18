import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm

def analyzeDOMPET():
    dataDict = {"catchment":[],"dompetSlope":[],"dompetMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)

        cat = u_getCatchmentName(file)

        waterYears = np.unique(df[waterYearVar])

        # loop through every water year and calculate the day of mean evapotranspiration
        dompets = []
        for waterYear in waterYears:
            ldf = df[df[waterYearVar] == waterYear]
            petForWaterYear = np.array(ldf[petVar])
            dompet = u_getDayOfMeanMagnitude(petForWaterYear)
            dompets.append(dompet)

        slope = u_regressionFunction(waterYears, dompets)
        mean = np.mean(dompets)

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["dompetSlope"].append(slope)
        dataDict["dompetMean"].append(mean)

        loop.set_description("Computing days of mean potential evapotranspiration")
        loop.update(1)


    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_dompet.csv")
    outDf.to_csv(outPath, index=False)

    loop.close()
