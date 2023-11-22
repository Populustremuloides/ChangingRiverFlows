import os
import pandas as pd
from data.metadata import *
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np

def addVar(row, var, dataDict, imputedTag):
    dataDict["streamflow variable"].append(predictablesToPretty[var])
    dataDict["change"].append(row[var])
    dataDict["imputed"].append(imputedTag)

    return dataDict


def plotPercents():

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    df = pd.read_csv(dataPath)

    # plot the scores *************************************
    dataDict = {"streamflow variable":[], "change":[], "imputed":[]}
    dataDictPercent = {"streamflow variable":[], "percent change":[], "imputed":[]}

    for index, row in df.iterrows():
        dataDict = addVar(row, "d_pPercentChange", dataDict, "measured")
        dataDict = addVar(row, "masdPercentChange", dataDict, "measured")

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedAll.csv")
    dfImputed = pd.read_csv(dataPath)

    for index, row in dfImputed.iterrows():
        dataDict = addVar(row, "d_pPercentChange", dataDict, "imputed + measured")
        dataDict = addVar(row, "masdPercentChange", dataDict, "imputed + measured")


    plotDf = pd.DataFrame.from_dict(dataDict)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(ax=ax, data=plotDf, x="streamflow variable",y="change", hue="imputed", split=True)
    plt.ylim(-20,20)
    plt.title("Distribution of Changes in Runoff Ratio")
    plt.ylabel("% change / year")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(figurePath, "distributionsPercents.png"))
    plt.clf()


def plotRunoffRatio():

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    df = pd.read_csv(dataPath)

    # plot the scores *************************************
    dataDict = {"streamflow variable":[], "change":[], "imputed":[]}
    dataDictPercent = {"streamflow variable":[], "percent change":[], "imputed":[]}

    for index, row in df.iterrows():
        dataDict = addVar(row, "d_pSlope", dataDict, "measured")

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedAll.csv")
    dfImputed = pd.read_csv(dataPath)

    for index, row in dfImputed.iterrows():
        dataDict = addVar(row, "d_pSlope", dataDict, "imputed + measured")


    plotDf = pd.DataFrame.from_dict(dataDict)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(ax=ax, data=plotDf, x="streamflow variable",y="change", hue="imputed", split=True)
    plt.ylim(-0.05, 0.05)
    plt.title("Distribution of Changes in Runoff Ratio")
    plt.ylabel("change in Ratio / year")
    plt.xticks(rotation=0)
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

    dataPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedAll.csv")
    dfImputed = pd.read_csv(dataPath)

    for index, row in dfImputed.iterrows():
        dataDict = addVar(row, "domfSlope", dataDict, "imputed + measured")
        dataDict = addVar(row, "dopfSlope", dataDict, "imputed + measured")
        dataDict = addVar(row, "pommfSlope", dataDict, "imputed + measured")


    plotDf = pd.DataFrame.from_dict(dataDict)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(ax=ax, data=plotDf, x="streamflow variable",y="change", hue="imputed", split=True)
    plt.ylim(-20,20)
    plt.title("Distribution of Changes in Timing and Periodicity of Flow")
    plt.ylabel("change in days / year")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(figurePath, "distributionsDays.png"))
    plt.clf()

def plotDistributions():
    plotDays()
    plotVolume()
    plotRunoffRatio()
    plotPercents()
