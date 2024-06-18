import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm
import matplotlib as mpl


import matplotlib.pyplot as plt

def calculateBudgetDeficits():
    '''
    Calculates the error between inputs and outputs: P - ET - Q
    '''

    dataDict = {"catchment":[],"budget_deficit":[], "percent_deficit":[], "catchment_area":[],"numWaterYears":[], "precipMean":[], "etMean":[],"dischargeMean":[]}

    numCats = len(os.listdir(augmentedTimeseriesPath))
    loop = tqdm(total=numCats)
    
    metaDf = pd.read_csv(metadataPath)
    metaDf["Catchment ID"] = metaDf["Catchment ID"].astype(str)
    
    countries = []
    for file in os.listdir(augmentedTimeseriesPath):
        dataFilePath = os.path.join(augmentedTimeseriesPath, file)
        df = pd.read_csv(dataFilePath)
        
        cat = u_getCatchmentName(file)

        lmDf = metaDf[metaDf["Catchment ID"] == cat]
        try:
            catchmentArea = lmDf["Catchment Area"].item()
        except:
            catchmentArea = list(lmDf["Catchment Area"])[0]
        try:
            countries.append(lmDf["Country"].item())
        except:
            countries.append("NONE")

        df = df.groupby(waterYearVar).mean()

        #deficits = df[precipVar] - df[etVar] - df[dischargeVar]
        #percentDeficits = 100 * (deficits / df[precipVar])

        waterYears = list(df.index)

        #meanDeficits = np.mean(deficits)
        #meanPercents = np.mean(percentDeficits)
        
        # total change in water budget
        etMean = np.mean(df[etVar]) 
        precipMean = np.mean(df[precipVar]) 
        dischargeMean = np.mean(df[dischargeVar]) 
        
        # calculate the budget imbalances
        deficit = (precipMean - etMean - dischargeMean)
        percentDeficit = 100 * (deficit / precipMean)
        
        # normalize to be yearly
        deficit = deficit / float(len(waterYears))
        percentDeficit = percentDeficit / float(len(waterYears))

        # store the newly harvested data
        dataDict["catchment"].append(cat)
        dataDict["budget_deficit"].append(deficit)
        dataDict["percent_deficit"].append(percentDeficit)
        dataDict["catchment_area"].append(catchmentArea)
        dataDict["numWaterYears"].append(len(waterYears))
        dataDict["precipMean"].append(precipMean)
        dataDict["etMean"].append(etMean)
        dataDict["dischargeMean"].append(dischargeMean)

        loop.set_description("Computing budget deficits")
        loop.update(1)
    
    countries = np.array(countries)
    australianCatchments = countries == "AU"

    # save the newly harvested data
    outDf = pd.DataFrame.from_dict(dataDict)

    aDf = outDf[australianCatchments] 
    plt.hist(aDf["precipMean"], alpha=0.5)
    plt.hist(aDf["etMean"] + aDf["dischargeMean"], alpha=0.5)
    #plt.hist(aDf["dischargeMean"], alpha=0.5)
    plt.xscale("log")
    plt.savefig(os.path.join(figurePath, "Australia.png"))
    print(aDf)
    quit()
    outPath = os.path.join(outputFilesPath, "timeseriesSummary_deficits.csv")
    outDf.to_csv(outPath, index=False)
   
    with open(os.path.join(logPath, "log_budgetDeficits.txt"), "w+") as logFile:
        logFile.writelines("The median % budget deficit / year value was: " + str(np.median(outDf["percent_deficit"][~outDf["percent_deficit"].isna()])) + "\n")
        logFile.writelines("The median budget deficit / year value was: " + str(np.median(outDf["budget_deficit"][~outDf["percent_deficit"].isna()])) + "\n")
            
        
        # calculate the budget deficit multiplied by catchment area
        mask = np.array(~outDf["percent_deficit"].isna())
        positiveMask = np.array(outDf["percent_deficit"] > 0)
        negativeMask = np.array(outDf["percent_deficit"] < 0)
        positiveMask = np.logical_and(mask, positiveMask)
        negativeMask = np.logical_and(mask, negativeMask)
        
        areas = outDf["catchment_area"].to_numpy()
        percentDeficits = outDf["percent_deficit"].to_numpy()
        
        logFile.writelines("number of catchments with postiive annual deficit: " + str(np.sum(positiveMask)) + "\n")
        logFile.writelines("number of catchments with negative annual deficit: " + str(np.sum(negativeMask)) + "\n")
        logFile.writelines("total area with direct budget deficit measurements available " + str(np.sum(areas[mask])) + "\n")

        logFile.writelines("% area with positive deficit: " + str(100. * (np.sum(areas[positiveMask]) / np.sum(areas[mask]))) + "\n")
        logFile.writelines("% area with negative deficit: " + str(100. * (np.sum(areas[negativeMask]) / np.sum(areas[mask]))) + "\n")
        logFile.writelines("median value for catchments with positive annual deficit: " + str(np.median(percentDeficits[positiveMask])) + "\n")
        logFile.writelines("weighted mean value for catchments with positive annual deficit: " + str(np.sum(percentDeficits[positiveMask] * areas[positiveMask]) / np.sum(areas[positiveMask])) + "\n")
        logFile.writelines("median value for catchments with negative annual deficit: " + str(np.median(percentDeficits[negativeMask])) + "\n")
        logFile.writelines("weighted mean value for catchments with negative annual deficit: " + str(np.sum(percentDeficits[negativeMask] * areas[negativeMask]) / np.sum(areas[negativeMask])) + "\n")

    loop.close()
