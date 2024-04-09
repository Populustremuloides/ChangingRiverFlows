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

varToTitle = {
        "masdMean":"Mean Annual Specific Dicharge (L/d/km$^2$)",
        "masdSlope":"Change in Mean Annual Specific Discharge $\Delta$(L/d/km$^2$) / year",
        "masdPercentChange":"Percent Change in Mean Annual Specific Discharge",
        "domfMean":"Day of Mean Flow",
        "domfSlope":"Change in Day of Mean Flow (days / year)",
        "dopfMean":"Day of Peak Flow (days)",
        "dopfSlope":"Change in Day of Peak Flow (days /year)",
        "pommfMean":"Period of Mean Flow (days)",
        "pommfSlope":"Change in Period of Mean Flow (days / year)",
        "d_pMean":"Runoff Ratio",
        "d_pSlope":"Change in Runoff Ratio per Year",
        "d_pPercentChange":"Percent Change in Runoff Ratio per Year",
        "m":"Fuh's Parameter",
        "budget_deficit":"Budget Deficit (Liters)",
        "percent_deficit":"% Budget Deficit",
        "cls5":"Proportion Shrubs",
        "cls3":"Proportion Deciduous Broadleaf Cover",
        "meanPercentDC_ModeratelyWell":"Percent Moderately Well-draining Soil"
        }



def getRegressors(ldf, udf, var1, var2):
    xl = np.expand_dims(ldf[var1].to_numpy(), axis=1)
    yl = ldf[var2].to_numpy()
    xu = np.expand_dims(udf[var1].to_numpy(), axis=1)
    yu = udf[var2].to_numpy()
    
    reg1 = TheilSenRegressor(random_state=0).fit(xl, yl)
    reg2 = TheilSenRegressor(random_state=0).fit(xu, yu)
    
    return reg1, reg2


def getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations):

    # determine lengths of lower/upper bound datasets
    ldf = df[df[var3] < threshold]
    udf = df[df[var3] >= threshold]
    lenLower = len(ldf[ldf.columns[0]])
    lenUpper = len(udf[udf.columns[0]])

    # determine real differnece in slopes
    reg1, reg2 = getRegressors(ldf, udf, var1, var2)
    realDiff = abs(reg1.coef_[0] - reg2.coef_[0])

    # randomly sample the df according to the sizes used in getSensitivity
    diffs = np.zeros(int(numIterations))
    indices = np.arange(len(df[df.columns[0]]))
    loop = tqdm(total=numIterations)
    for i in range(int(numIterations)):
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
    N, bins, patches = axs[1].hist(diffs, bins=100, density=False)
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
    axs[1].legend(handles, labels)
    axs[1].vlines(x=realDiff, ymin=0, ymax=maxHeight, color="k")
    #ax.set_xlabel("absolute difference in slope of\n\"" + predictablesToPretty[var2].replace("\n"," ") + "\"\nto\n\"" + predictorsToPretty[var1] + "\"", fontsize=11) 
    axs[1].set_xlabel("absolute difference in slope given a random selection of catchments", fontsize=11) 
    axs[1].set_ylabel("count", fontsize=11)
    #ax.set_title("Random Differences in Slopes Compared to Thresholding by\n\"" + predictorsToPretty[var3] + "\"")
    #axs[1].set_title("Empirical Probability Distribution of Differences in Slopes")
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(figurePath, "sensitivity_plot_" + str(var2) + "_" + str(var1) + "_" + str(var3) + ".png"), dpi=300)
    #plt.show()
    plt.clf()
    plt.close()
    
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

def plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=0, upperBound=1):

    sortIndices = np.argsort(df[var3])
    #colors = getColors
    #m = getM(variable=var3, cmap="seismic", df=df)
    #colors = getColors(var3, m, df)
    
    norm = plt.Normalize(vmin=lowerBound, vmax=upperBound)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
    scatter = axs[0].scatter(x=np.array(df[var1])[sortIndices], y=np.array(df[var2])[sortIndices], c=np.array(df[var3])[sortIndices], cmap="seismic", norm=norm)
    cbar = fig.colorbar(scatter, ax=axs[0], orientation="vertical")

    cbar.set_label(varToTitle[var3], fontsize=13)
    cbar.ax.tick_params(labelsize=13)
    legend = plt.legend()
    for label in legend.get_texts():
        label.set_fontsize(13)  # Set the desired fontsize (e.g., 12)

        # Adjust the marker size for legend handles (icons)
        for handle in legend.legendHandles:
                handle.set_sizes([40])
 

    if var1 in predictablesToPretty.keys():
        axs[0].set_xlabel(predictablesToPretty[var1])
    elif var1 in predictorsToPretty.keys():
        axs[0].set_xlabel(predictorsToPretty[var1])
    elif var1 in predictorsToPrettyPCA.keys():
        axs[0].set_xlabel(predictorsToPrettyPCA[var1])
    else:
        axs[0].set_xlabel(var1)

    if var2 in predictablesToPretty.keys():
        ylabel = predictablesToPretty[var2]
    elif var2 in predictorsToPretty.keys():
        ylabel = predictorsToPretty[var2]
    elif var2 in predictorsToPrettyPCA.keys():
        ylabel = predictorsToPrettyPCA[var2]
    axs[0].set_ylabel(ylabel.replace("\n"," "))
    
    if log:
        axs[0].yscale("log")
        axs[0].xscale("log")
    
    var1Range = np.expand_dims(np.linspace(np.min(df[var1]), np.max(df[var1])), axis=1)
    lowerPreds = reg1.predict(var1Range)
    upperPreds = reg2.predict(var1Range)
    
    axs[0].plot(var1Range, lowerPreds, c="b", label=predictorsToPretty[var3] + " < " + str(threshold))
    axs[0].plot(var1Range, upperPreds, c="r", label=predictorsToPretty[var3] + " >= " + str(threshold))
    axs[0].legend()
    axs[0].set_ylim(np.min(df[var2]) - (np.std(df[var2]) / 3.), np.max(df[var2]) + - (np.std(df[var2]) / 3.))
    #axs[0].tight_layout()
    axs[0].grid()

    return fig, axs
    #plt.savefig(os.path.join(figurePath, str(var2) + "_" + str(var1) + "_" + str(var3) + ".png"))
    #plt.clf()
    #plt.close()


def writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval):
        slopeU = slopeU[0]
        slopeL = slopeL[0]

        oFile.writelines("var1: " + str(var1) + ": " + predictorsToPretty[var1].replace("\n", " ") + "\n")
        oFile.writelines("var2: " + str(var2) + ": " + predictablesToPretty[var2].replace("\n"," ") + "\n")
        oFile.writelines("var3: " + str(var3) + ": " + predictorsToPretty[var3].replace("\n"," ") + "\n")
        oFile.writelines("threshold: " + str(threshold) + "\n")
        oFile.writelines("p-value of the slopes being this different: " + str(pval) + "\n")
        oFile.writelines("slope of lower group: " + str(slopeL) + "\n")
        oFile.writelines("slope of upper group: " + str(slopeU) + "\n")
        oFile.writelines("lower group / upper group: " + str(slopeL / slopeU) + "\n")
        oFile.writelines("upper group / lower group: " + str(slopeU / slopeL) + "\n")
        oFile.writelines("(lower group - upper group) / lower group: " + str((slopeL - slopeU)/ slopeL) + "\n")
        oFile.writelines("(lower group - upper group) / upper group: " + str((slopeL - slopeU)/ slopeU) + "\n")
        oFile.writelines("(upper group - lower group) / lower group: " + str((slopeU - slopeL)/ slopeL) + "\n")
        oFile.writelines("(upper group - lower group) / upper group: " + str((slopeU - slopeL)/ slopeU) + "\n")
        oFile.writelines("coefficient of determination of lower group: " + str(rSquaredL) + "\n")
        oFile.writelines("coefficient of determination of upper group: " + str(rSquaredU) + "\n")
        oFile.writelines("***************************************\n\n")

def getSingle(df, var1, var2, normalize=True):
    d1 = np.array(df[var1])
    d2 = np.array(df[var2])
    d3 = np.ones_like(d2) 
    data = np.array([d2]).T

    # normalize 
    if normalize:
        data = data - np.mean(data, axis=0)
        data = data / np.std(data, axis=0)

    d3 = np.expand_dims(d3, axis=0).T
    data = np.concatenate((d3, data), axis=1)
    
    if normalize:
        d1 = d1 - np.mean(d1)
        d1 = d1 / np.std(d1)

    reg = TheilSenRegressor().fit(data, d1) 

    return reg.coef_, reg.score(data, d1)
   

def getInteraction(df, var1, var2, var3, normalize=True):
    d1 = np.array(df[var1])
    d2 = np.array(df[var2])
    d3 = np.array(df[var3])
    d4 = np.ones_like(d3) 
    data = np.array([d2, d2 * d3, d3]).T

    # normalize 
    if normalize:
        data = data - np.mean(data, axis=0)
        data = data / np.std(data, axis=0)

    d4 = np.expand_dims(d4, axis=0).T
    data = np.concatenate((d4, data), axis=1)
    
    if normalize:
        d1 = d1 - np.mean(d1)
        d1 = d1 / np.std(d1)

    reg = TheilSenRegressor().fit(data, d1) 

    return reg.coef_, reg.score(data, d1)

def calculateSensitivities2(numIterations=1e2):
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(dataFilePath)
    df = df[df["domfSlope"] < 30] # remove a single outlier
    df = df[~df["pommfSlope"].isna()]
    df["potentialFlow"] = df["maspMean"] - df["masetMean"]
    df["logArea"] = np.log10(np.array(df["Catchment Area"]) + 1)
    df["cls1_100"] = 100 * df["cls1"]
    df["cls3_100"] = 100 * df["cls3"]
    df["cls5_100"] = 100 * df["cls5"]
    df["forest_100"] = 100 * df["forest"]
    
    with open(os.path.join(logPath, "log_sensitivitiesAndCorrelations.txt"), "w+") as logFile:        
        coefficients, rSquared = getSingle(df, var1="forest_100", var2="logArea", normalize=False)
        logFile.write("correlation between log area and forest cover:\n")
        logFile.write("bias and log area coefficients: " + str(coefficients) + "\n")
        logFile.write("r-Squared value: " + str(rSquared))

    variables = [
            ["masdPercentChange", "maspPercentChange", "m"],
            ["p_petSlope", "pommfSlope","meanPercentDC_ModeratelyWell"],
            ["p_petSlope", "pommfSlope","meanPercentDC_ModeratelyWell"],
            ["domfSlope", "dompSlope", "cls5_100"],
            ["domfSlope", "dompSlope", "cls3_100"],
            ["domfSlope", "dompSlope", "cls1_100"],
            ["d_pPercentChange", "pet_pSlope", "m"],
            ["masdPercentChange", "maspPercentChange", "cls3_100"],
            ["masdPercentChange", "maspPercentChange", "cls1_100"],
            ["masdPercentChange", "maspPercentChange", "forest_100"],
            ["masdPercentChange", "maspPercentChange", "cls5_100"],
            ["d_pPercentChange", "maspPercentChange", "cls3_100"],
            ["potentialFlow","logArea","forest_100"],
            ["potentialFlow","logArea","Dam_Count"]
            ]

    outDict = {"var1_target":[], "var2":[],"var3":[],"intercept":[], "var2Slope":[],"interactionSlope":[],"var3Slope":[], "r_squared":[]} 
    for var1, var2, var3 in variables:
        normalize = False # decided to not normalize any inputs, but keeping the option for code flexibility
        coefficients, rSquared = getInteraction(df, var1, var2, var3, normalize)
        outDict["var1_target"].append(var1)
        outDict["var2"].append(var2)
        outDict["var3"].append(var3)
        outDict["intercept"].append(coefficients[0])
        outDict["var2Slope"].append(coefficients[1])
        outDict["interactionSlope"].append(coefficients[2])
        outDict["var3Slope"].append(coefficients[3])
        outDict["r_squared"].append(rSquared)

    outDf = pd.DataFrame.from_dict(outDict)
    outDf.to_csv(os.path.join(outputFilesPath, "interactions.csv"), index=False)

