import os
import pandas as pd
from data.metadata import *
from analyses.colorCatchments import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
from sklearn.linear_model import TheilSenRegressor
import copy
import math
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def getRegressors(ldf, udf, var1, var2):
    xl = np.expand_dims(ldf[var1].to_numpy(), axis=1)
    yl = ldf[var2].to_numpy()
    xu = np.expand_dims(udf[var1].to_numpy(), axis=1)
    yu = udf[var2].to_numpy()
    
    reg1 = TheilSenRegressor().fit(xl, yl)
    reg2 = TheilSenRegressor().fit(xu, yu)
    
    return reg1, reg2


def getPVal(df, var1, var2, var3, threshold):

    # determine lengths of lower/upper bound datasets
    numIterations = 1000
    ldf = df[df[var3] < threshold]
    udf = df[df[var3] >= threshold]
    lenLower = len(ldf[ldf.columns[0]])
    lenUpper = len(udf[udf.columns[0]])

    # determine real differnece in slopes
    reg1, reg2 = getRegressors(ldf, udf, var1, var2)
    realDiff = abs(reg1.coef_[0] - reg2.coef_[0])

    # randomly sample the df according to the sizes used in getSensitivity
    diffs = np.zeros(numIterations)
    indices = np.arange(len(df[df.columns[0]]))
    loop = tqdm(total=numIterations)
    for i in range(numIterations):
        np.random.shuffle(indices) # shuffle the indices
        lowerMask = indices < lenLower
        ldf = df[lowerMask]
        udf = df[~lowerMask]       

        reg1, reg2 = getRegressors(ldf, udf, var1, var2)
        diff = abs(reg1.coef_[0] - reg2.coef_[0])
        diffs[i] = diff
        loop.set_description("calculating p-value")
        loop.update(1)
    
    # create a histogram using those outputs
    N, bins, patches = plt.hist(diffs, bins=100, density=False)
    maxHeight = np.max(N)

    diffs = np.sort(diffs)
    numTo95 = int(diffs.shape[0] * 0.95)
    numTo975 = int(diffs.shape[0] * 0.975)

    threshold95 = diffs[numTo95]
    threshold975 = diffs[numTo975]
    
    # color the various sections
    for i, bini in enumerate(bins[1:]):
        if bini < threshold95:
            patches[i].set_facecolor("b")
        elif bini < threshold975:
            patches[i].set_facecolor("orange")
        else:
            patches[i].set_facecolor("r")

    # create the legend
    handles = [
                Rectangle((0, 0), 1, 1, color='b', ec='k'),
                Rectangle((0, 0), 1, 1, color='orange', ec='k'),
                Rectangle((0, 0), 1, 1, color='r', ec='k'),
                #Line2D([0], [0], color='blue', linestyle='-'),
                # Add Line2D for the vlines symbol
                Line2D([0], [0], color='black', marker='|', markersize=10, linestyle='None', label='vlines')
            ]

    labels= ["<95%","<97.5%","<100%","actual absolute difference"]
    plt.legend(handles, labels)
    plt.vlines(x=realDiff, ymin=0, ymax=maxHeight, color="k")
    plt.xlabel("absolute difference in slope of\n\"" + predictablesToPretty[var2].replace("\n","") + "\"\nto\n\"" + predictorsToPretty[var1] + "\"", fontsize=11) 
    plt.ylabel("count", fontsize=11)
    plt.title("Random Differences in Slopes Compared to Thresholding by\n\"" + predictorsToPretty[var3] + "\"")
    plt.gca().spines["top"].set_visible(False)  
    plt.tight_layout()
    plt.gca().spines["right"].set_visible(False)
    plt.savefig(os.path.join(figurePath, "p_value_plot_" + str(var2) + "_" + str(var1) + "_" + str(var3) + ".png"))
    plt.clf()
    #plt.close()
    
    # calculate the p-value
    numGreater = float(np.sum(diffs > realDiff))
    pval = numGreater / float(numIterations)

    return pval

def getSensitivity(df, var1, var2, var3, threshold):
    ldf = df[df[var3] < threshold]
    udf = df[df[var3] >= threshold]
    
    reg1, reg2 = getRegressors(ldf, udf, var1, var2)

    xl = np.expand_dims(ldf[var1].to_numpy(), axis=1)
    yl = ldf[var2].to_numpy()
    xu = np.expand_dims(udf[var1].to_numpy(), axis=1)
    yu = udf[var2].to_numpy()
 
    coefDetermination1 = reg1.score(xl, yl)
    coefDetermination2 = reg2.score(xu, yu)

    return reg1.coef_, reg2.coef_, coefDetermination1, coefDetermination2, reg1, reg2

def plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False):

    sortIndices = np.argsort(df[var3])
    colors = getColors
    m = getM(variable=var3, cmap="seismic", df=df)
    colors = getColors(var3, m, df)

    plt.scatter(x=np.array(df[var1])[sortIndices], y=np.array(df[var2])[sortIndices], c=np.array(colors)[sortIndices])
    
    if var1 in predictablesToPretty.keys():
        plt.xlabel(predictablesToPretty[var1])
    elif var1 in predictorsToPretty.keys():
        plt.xlabel(predictorsToPretty[var1])
    elif var1 in predictorsToPrettyPCA.keys():
        plt.xlabel(predictorsToPrettyPCA[var1])
    else:
        plt.xlabel(var1)

    if var2 in predictablesToPretty.keys():
        ylabel = predictablesToPretty[var2]
    elif var2 in predictorsToPretty.keys():
        ylabel = predictorsToPretty[var2]
    elif var2 in predictorsToPrettyPCA.keys():
        ylabel = predictorsToPrettyPCA[var2]
    plt.ylabel(ylabel)
    
    if log:
        plt.yscale("log")
        plt.xscale("log")
    
    var1Range = np.expand_dims(np.linspace(np.min(df[var1]), np.max(df[var1])), axis=1)
    lowerPreds = reg1.predict(var1Range)
    upperPreds = reg2.predict(var1Range)
    
    plt.plot(var1Range, lowerPreds, c="b", label=predictorsToPretty[var3] + " < " + str(threshold))
    plt.plot(var1Range, upperPreds, c="r", label=predictorsToPretty[var3] + " >= " + str(threshold))
    plt.legend()
    plt.ylim(np.min(df[var2]) - (np.std(df[var2]) / 3.), np.max(df[var2]) + - (np.std(df[var2]) / 3.))
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, str(var2) + "_" + str(var1) + "_" + str(var3) + ".png"))
    plt.clf() 


def writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval):
        oFile.writelines("var1: " + str(var1) + ": " + predictorsToPretty[var1] + "\n")
        oFile.writelines("var2: " + str(var2) + ": " + predictablesToPretty[var2] + "\n")
        oFile.writelines("var3: " + str(var3) + ": " + predictorsToPretty[var3] + "\n")
        oFile.writelines("threshold: " + str(threshold) + "\n")
        oFile.writelines("p-value of the slopes being this different: " + str(pval) + "\n")
        oFile.writelines("slope of lower group: " + str(slopeL) + "\n")
        oFile.writelines("slope of upper group: " + str(slopeU) + "\n")
        oFile.writelines("coefficient of determination of lower group: " + str(rSquaredL) + "\n")
        oFile.writelines("coefficient of determination of upper group: " + str(rSquaredU) + "\n")
        oFile.writelines("***************************************\n\n")


def calculateSensitivities():
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(dataFilePath)
    df = df[~df["d_pSlope"].isna()]
    df = df[df["pet_pSlope"] > -0.2] # remove outliers
    df = df[df["dompSlope"] < 10] # remove outliers
    df = df.dropna()

    with open(os.path.join(outputFilesPath, "sensitivities.txt"), "w+") as oFile:
        var1 = "p_petSlope"
        var2 = "pommfSlope"
        var3 = "meanPercentDC_ModeratelyWell"
        threshold = 0.5
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        pval = getPVal(df, var1, var2, var3, threshold)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False)

        var1 = "dompSlope"
        var2 = "domfSlope"
        var3 = "cls5"
        threshold = 0.05
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        pval = getPVal(df, var1, var2, var3, threshold)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False)

        var1 = "dompSlope"
        var2 = "domfSlope"
        var3 = "cls3"
        threshold = 0.05
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        pval = getPVal(df, var1, var2, var3, threshold)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False)

        var1 = "pet_pSlope"
        var2 = "d_pPercentChange"
        var3 = "m"
        threshold = 6
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        pval = getPVal(df, var1, var2, var3, threshold)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False)


        var1 = "maspPercentChange"
        var2 = "d_pPercentChange"
        var3 = "cls3"
        threshold = 0.05
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        pval = getPVal(df, var1, var2, var3, threshold)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False)
