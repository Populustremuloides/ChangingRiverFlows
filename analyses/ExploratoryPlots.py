import os
import pandas as pd
from data.metadata import *
from analyses.colorCatchments import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from colorCatchments import getColors
from scipy.stats import spearmanr
import copy

def plotVar(df, var1, var2, colorVar="m", lowerBound=0, upperBound=1, log=True, tag="real_values", abso=False):
    sortIndices = np.argsort(df[colorVar])

    colors = np.array(df[colorVar])
    lowerMask = colors < lowerBound
    upperMask = colors > upperBound
    colors[upperMask] = upperBound
    colors[lowerMask] = lowerBound
    percentTruncated = 100. * ((np.sum(upperMask) + np.sum(lowerMask)) / lowerMask.shape[0])
    
    fig, ax = plt.subplots()

    if abso:
        colors = ax.scatter(x=np.array(df[var1])[sortIndices], y=np.array(np.abs(df[var2]))[sortIndices], c=np.array(colors)[sortIndices], cmap="seismic")
    else:
        colors = ax.scatter(x=np.array(df[var1])[sortIndices], y=np.array(df[var2])[sortIndices], c=np.array(colors)[sortIndices], cmap="seismic")
    cbar = fig.colorbar(colors, ax=ax, orientation="vertical") 
    cbar.set_label(predictorsToPretty[colorVar].replace("\n"," "))

    if var1 in predictablesToPretty.keys():
        ax.set_xlabel(predictablesToPretty[var1])
    elif var1 in predictorsToPretty.keys():
        ax.set_xlabel(predictorsToPretty[var1])
    elif var1 in predictorsToPrettyPCA.keys():
        ax.set_xlabel(predictorsToPrettyPCA[var1])
    else:
        ax.set_xlabel(var1)

    if var2 in predictablesToPretty.keys():
        ylabel = predictablesToPretty[var2]
    elif var2 in predictorsToPretty.keys():
        ylabel = predictorsToPretty[var2]
    elif var2 in predictorsToPrettyPCA.keys():
        ylabel = predictorsToPrettyPCA[var2]
    if abso:
        ylabel = "absolute value of\n" + ylabel
    ax.set_ylabel(ylabel)

    if log:
        plt.yscale("log")
        plt.xscale("log")

    plt.tight_layout()
    outPath = os.path.join(individualVarsPath, tag)
    if not os.path.exists(outPath):
        os.mkdir(outPath)
    plt.savefig(os.path.join(outPath, str(var2) + "_" + str(var1) + ".png"))
    plt.clf()
    plt.close()


def exploratoryPlots(colorVar="m", lowerBound=0, upperBound=1):
    ''' plot the top numToPlot predictors for each variable '''

    numToPlot = 5
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(dataFilePath)
    df = df[df["domfSlope"] < 30]

    for predictable in list(predictablesToPretty.keys()):
        ldf = pd.read_csv(os.path.join(outputFilesPath, "individualCorrs_" + str(predictable) + ".csv"))
        for i in range(numToPlot):
            var = ldf["predictors"][i]
            plotVar(df, var, predictable, colorVar, lowerBound=lowerBound, upperBound=upperBound, log=False, tag="real_values", abso=False)


    for predictable in list(predictablesToPretty.keys()):
        ldf = pd.read_csv(os.path.join(outputFilesPath, "individualCorrs_abs_" + str(predictable) + ".csv"))
        for i in range(numToPlot):
            var = ldf["absolute_predictors"][i]
            plotVar(df, var, predictable, colorVar, lowerBound=lowerBound, upperBound=upperBound, log=False, tag="absolute_values", abso=True)



