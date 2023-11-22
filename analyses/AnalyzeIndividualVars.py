import os
import pandas as pd
from data.metadata import *
from analyses.colorCatchments import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.stats import spearmanr
import copy

def plotFuh():
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(dataFilePath)

    colorVar = "d_pSlope"
    #cmap = "seismic"
    xVar = "p_petMean"
    yVar = "d_pMean"
    
    p_pets = np.linspace(0, np.max(df["p_petMean"]) + 0.5, 100)
    for w in [1, 90]:
        ys = np.power(1 + np.power(p_pets, -1 * w), 1. / w)  - np.power(p_pets, -1)
        if w == 1:
            plt.plot(p_pets, ys, c="k", linestyle="--", label="m=1")
        else:
            plt.plot(p_pets, ys, c="k", linestyle="-", label="m=$\infty$")

    #m = getM(colorVar, cmap, df)
    #cs = getColors(colorVar, m, df, transform=None)
    plt.scatter(x=df[xVar], y=df[yVar], alpha=0.15) #, c=cs, alpha=0.5)
    plt.ylim(-0.1, 1.2)
    plt.xlabel(predictorsToPretty[xVar])
    plt.ylabel(predictorsToPretty[yVar])
    plt.legend()
    plt.savefig(os.path.join(figurePath, "fuhs_equation.png"))
    plt.clf() 

    plt.hist(df["m"], density=True, bins=50)
    plt.title("Distribution of Fuh's Parameter")
    plt.xlabel("m value")
    plt.ylabel("density")
    plt.savefig(os.path.join(figurePath, "fuhs_distribution.png"))
    plt.clf() 



def testVar(df, v1, v2, oFile, abso=False):
    x = df[v1]
    if abso == True:
        y = np.abs(df[v2])
    else:
        y = df[v2]
    result = spearmanr(x, y)

    oFile.writelines("Results for analysis of " + str(v1) + " and " + str(v2) + ":\n")
    if abso:
        oFile.writelines("correlation between " + str(v1) + " and abs(" + str(v2) + "): " + str(result[0]) + "\n")
    else:
        oFile.writelines("correlation between " + str(v1) + " and " + str(v2) + ": " + str(result[0]) + "\n")

    oFile.writelines("p-value: " + str(result[1]) + "\n\n\n")

def newLine(oFile):
    oFile.writelines("***************************************\n\n")

def plotWChanges(df):
    pass

def testAll(df, var1, oFile, abso):
    testVar(df, var1, "d_pSlope", oFile, abso=True)
    testVar(df, var1, "masdSlope", oFile, abso=True)
    testVar(df, var1, "domfSlope", oFile, abso=True)
    testVar(df, var1, "dopfSlope", oFile, abso=True)
    testVar(df, var1, "pommfSlope", oFile, abso=True)
    newLine(oFile) 

def analyzeCorrelations():
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(dataFilePath)
    df = df[~df["d_pSlope"].isna()]

    absos = [True, False]

    for abso in absos:
        for predictable in list(predictablesToPretty.keys()):
            ldf = copy.copy(df[[predictable] + list(predictorsToPretty.keys())])
            ldf = ldf.dropna()
            cols = np.array(ldf.columns)
            if abso:
                ldf[predictable] = ldf[predictable].abs()
            result = spearmanr(ldf.to_numpy())
            correlations = result[0][0,:]
            pvals = result[1][0,:]
            indices = np.flip(np.argsort(np.abs(correlations)))

            if abso:
                outDf = pd.DataFrame.from_dict({"absolute_predictors":cols[indices[1:]],"correlations":correlations[indices[1:]],"pvals":pvals[indices[1:]]})
                outDf.to_csv(os.path.join(outputFilesPath, "individualCorrs_abs_" + str(predictable) + ".csv"), index=False)
            else:
                outDf = pd.DataFrame.from_dict({"predictors":cols[indices[1:]],"correlations":correlations[indices[1:]],"pvals":pvals[indices[1:]]})
                outDf.to_csv(os.path.join(outputFilesPath, "individualCorrs_" + str(predictable) + ".csv"), index=False)

    '''
    with open(os.path.join(outputFilesPath, "individual_correlations.txt"), "w+") as oFile:

        # use just the imputed data

        testAll(df, "m", oFile, abso=True)
        testAll(df, "p_petSlope", oFile, abso=True)
        testAll(df, "pet_etSlope", oFile, abso=True)
        testAll(df, "human", oFile, abso=True)
        testAll(df, "forest", oFile, abso=True)
        testAll(df, "cls7", oFile, abso=True)
        testAll(df, "Dam_Count", oFile, abso=True)
        testAll(df, "Dam_SurfaceArea", oFile, abso=True)
        testAll(df, "pet_etMean", oFile, abso=True)
        testAll(df, "Catchment Area", oFile, abso=True)

        # switch to the PCA data
        dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedPCA.csv")
        df = pd.read_csv(dataFilePath)

        for i in range(1, 15):
            testAll(df, str(i), oFile, abso=True)
    '''
def plotVar(df, var1, var2, log=True):
    plt.scatter(x=df[var1], y=df[var2])
    
    if var1 in predictablesToPretty.keys():
        plt.xlabel(predictablesToPretty[var1])
    elif var1 in predictorsToPretty.keys():
        plt.xlabel(predictorsToPretty[var1])
    elif var1 in predictorsToPrettyPCA.keys():
        plt.xlabel(predictorsToPrettyPCA[var1])
    else:
        plt.xlabel(var1)

    if var2 in predictablesToPretty.keys():
        plt.ylabel(predictablesToPretty[var2])
    elif var2 in predictorsToPretty.keys():
        plt.ylabel(predictorsToPretty[var2])
    elif var2 in predictorsToPrettyPCA.keys():
        plt.ylabel(predictorsToPrettyPCA[var2])
    
    if log:
        plt.yscale("log")
        plt.xscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(individualVarsPath, str(var1) + "_" + str(var2) + ".png"))
    plt.clf() 


    
def plotRelations():
    ''' plot the top numToPlot predictors for each variable '''
    numToPlot = 5
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(dataFilePath)

    for predictable in list(predictablesToPretty.keys()):
        ldf = pd.read_csv(os.path.join(outputFilesPath, "individualCorrs_" + str(predictable) + ".csv"))
        for i in range(numToPlot):
            var = ldf["predictors"][i]
            plotVar(df, var, predictable)


    for predictable in list(predictablesToPretty.keys()):
        ldf = pd.read_csv(os.path.join(outputFilesPath, "individualCorrs_abs_" + str(predictable) + ".csv"))
        for i in range(numToPlot):
            var = ldf["absolute_predictors"][i]
            plotVar(df, var, predictable)



def analyzeIndividualVars():
    plotFuh()
    analyzeCorrelations()
    plotRelations()

