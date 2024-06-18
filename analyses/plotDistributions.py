import os
import pandas as pd
from data.metadata import *
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np


predictablesToPretty2 = {"masdPercentChange":"mean annual\nspecific discharge", 
                         "d_pPercentChange":"runoff ratio",
                         "percent_deficit":"water budget\nimbalance",
                         "dopfSlope":"day of\npeak flow",
                         "domfSlope":"day of\nmean flow",
                         "pommfSlope":"period of\nmean flow",
                         "d_pSlope":"runoff ratio",
                         "masdSlope":"mean annual\nspecific discharge",
                         "numWaterYears":"years of data"
                         }

def addVar(row, var, dataDict, imputedTag):
    dataDict["streamflow variable"].append(predictablesToPretty2[var])
    dataDict["change"].append(row[var])
    dataDict["imputed"].append(imputedTag)



def plotPercents():

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    df = pd.read_csv(dataPath)

    # plot the scores *************************************
    dataDict = {"streamflow variable":[], "change":[], "imputed":[]}
    dataDictPercent = {"streamflow variable":[], "percent change":[], "imputed":[]}

    for index, row in df.iterrows():
        dataDict = addVar(row, "d_pPercentChange", dataDict, "measured")
        dataDict = addVar(row, "masdPercentChange", dataDict, "measured")
        dataDict = addVar(row, "percent_deficit", dataDict, "measured")
    '''
    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedAll.csv")
    dfImputed = pd.read_csv(dataPath)

    for index, row in dfImputed.iterrows():
        dataDict = addVar(row, "d_pPercentChange", dataDict, "imputed + measured")
        dataDict = addVar(row, "masdPercentChange", dataDict, "imputed + measured")
    '''

    plotDf = pd.DataFrame.from_dict(dataDict)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    #sns.violinplot(ax=ax, data=plotDf, x="streamflow variable",y="change", hue="imputed", split=True)
    sns.boxplot(ax=ax, data=plotDf, x="streamflow variable", y="change", color="skyblue")
    plt.ylim(-15,15)
    #plt.title("Distribution of Changes in Runoff Rati")
    plt.ylabel("% change / year")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "distributionsPercents.png"))
    plt.clf()


def plotNumWaterYears():

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    df = pd.read_csv(dataPath)

    # plot the scores *************************************
    dataDict = {"streamflow variable":[], "change":[], "imputed":[]}
    dataDictPercent = {"streamflow variable":[], "percent change":[], "imputed":[]}

    for index, row in df.iterrows():
        dataDict = addVar(row, "numWaterYears", dataDict, "measured")

    plotDf = pd.DataFrame.from_dict(dataDict)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(2.6, 5))
    #sns.violinplot(ax=ax, data=plotDf, x="streamflow variable",y="change", hue="imputed", split=True)
    sns.boxplot(ax=ax, data=plotDf, x="streamflow variable", y="change", color="skyblue")
    plt.ylim(5,25)
    #plt.title("Distribution of Changes in Runoff Rati")
    plt.ylabel("number of complete years of data")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "distributionsWaterYears.png"))
    plt.clf()



def plotRunoffRatio():

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    df = pd.read_csv(dataPath)

    # plot the scores *************************************
    dataDict = {"streamflow variable":[], "change":[], "imputed":[]}
    dataDictPercent = {"streamflow variable":[], "percent change":[], "imputed":[]}

    for index, row in df.iterrows():
        dataDict = addVar(row, "d_pSlope", dataDict, "measured")
    
    '''
    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedAll.csv")
    dfImputed = pd.read_csv(dataPath)

    for index, row in dfImputed.iterrows():
        dataDict = addVar(row, "d_pSlope", dataDict, "imputed + measured")
    '''

    plotDf = pd.DataFrame.from_dict(dataDict)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(4, 5))
    #sns.violinplot(ax=ax, data=plotDf, x="streamflow variable",y="change", hue="imputed", split=True)
    sns.violinplot(ax=ax, data=plotDf, x="streamflow variable", y="change", color="skyblue")
    plt.ylim(-0.05, 0.05)
    plt.title("Distribution of Changes in Runoff Ratio")
    plt.ylabel("change in Ratio / year")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "distributionsRunoffRatio.png"))
    plt.clf()


def plotVolume():

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    df = pd.read_csv(dataPath)

    # plot the scores *************************************
    dataDict = {"streamflow variable":[], "change":[], "imputed":[]}
    dataDictPercent = {"streamflow variable":[], "percent change":[], "imputed":[]}

    for index, row in df.iterrows():
        dataDict = addVar(row, "masdSlope", dataDict, "measured")

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedAll.csv")
    dfImputed = pd.read_csv(dataPath)

    for index, row in dfImputed.iterrows():
        dataDict = addVar(row, "masdSlope", dataDict, "imputed + measured")


    plotDf = pd.DataFrame.from_dict(dataDict)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(ax=ax, data=plotDf, x="streamflow variable",y="change", hue="imputed", split=True)
    plt.ylim(-150000,150000)
    plt.title("Distribution of Changes in Volume of Flow")
    plt.ylabel("change in flow (L/day) / year")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "distributionsVolume.png"))
    plt.clf()


def plotDays():

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    df = pd.read_csv(dataPath)

    # plot the scores *************************************
    dataDict = {"streamflow variable":[], "change":[], "imputed":[]}
    dataDictPercent = {"streamflow variable":[], "percent change":[], "imputed":[]}

    for index, row in df.iterrows():
        dataDict = addVar(row, "domfSlope", dataDict, "measured")
        dataDict = addVar(row, "dopfSlope", dataDict, "measured")
        dataDict = addVar(row, "pommfSlope", dataDict, "measured")
    
    '''
    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedAll.csv")
    dfImputed = pd.read_csv(dataPath)

    for index, row in dfImputed.iterrows():
        dataDict = addVar(row, "domfSlope", dataDict, "imputed + measured")
        dataDict = addVar(row, "dopfSlope", dataDict, "imputed + measured")
        dataDict = addVar(row, "pommfSlope", dataDict, "imputed + measured")
    '''

    plotDf = pd.DataFrame.from_dict(dataDict)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    #sns.violinplot(ax=ax, data=plotDf, x="streamflow variable",y="change", hue="imputed", split=True)
    sns.boxplot(ax=ax, data=plotDf, x="streamflow variable",y="change", color="skyblue")

    plt.ylim(-15,15)
    #plt.title("Distribution of Changes in Timing and Periodicity of Flow")
    plt.ylabel("change in days / year")
    plt.xticks(rotation=0)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "distributionsDays.png"))
    plt.clf()
  
def getDistributionDescriptions():
    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    df = pd.read_csv(dataPath)
    
    variables = ["domfSlope", "dopfSlope", "pommfSlope", "masdSlope","masdPercentChange", "d_pSlope","percent_deficit","d_pPercentChange"]
    
    dataDict = {"variable":[],"median":[],"mean":[]}

    for var in variables:
        dataDict["variable"].append(var)
        dataDict["median"].append(np.nanmedian(df[var]))
        dataDict["mean"].append(np.nanmean(df[var]))

    summaryDf = pd.DataFrame.from_dict(dataDict)
    summaryDf.to_csv(os.path.join(outputFilesPath, "distributionSummaries.csv"), index=False)

def plotDistributions():
    getDistributionDescriptions()

    plotDays()
    plotVolume()
    plotRunoffRatio()
    plotPercents()
    plotNumWaterYears()
