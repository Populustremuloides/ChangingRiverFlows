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
        "masdMean":"Mean Annual Specific\nDicharge (L/d/km$^2$)",
        "masdSlope":"Change in Mean Annual\nSpecific Discharge $\Delta$(L/d/km$^2$) / year",
        "masdPercentChange":"Percent Change in\nMean Annual Specific Discharge",
        "domfMean":"Day of Mean Flow",
        "domfSlope":"Change in Day of\nMean Flow (days / year)",
        "dopfMean":"Day of Peak Flow (days)",
        "dopfSlope":"Change in Day of\nPeak Flow (days /year)",
        "pommfMean":"Period of Mean Flow (days)",
        "pommfSlope":"Change in Period of\nMean Flow (days / year)",
        "d_pMean":"Runoff Ratio",
        "d_pSlope":"Change in Runoff Ratio per Year",
        "d_pPercentChange":"Percent Change in\nRunoff Ratio per Year",
        "m":"Fuh's Parameter",
        "budget_deficit":"Budget Deficit (Liters)",
        "percent_deficit":"% Budget Deficit",
        "cls5_100":"Percent Shrubs",
        "cls3_100":"Percent Deciduous\nBroadleaf Cover",
        "meanPercentDC_ModeratelyWell":"Percent Moderately\nWell-draining Soil",
        "maspMean":"Mean Annual Specific\nPrecipitation (L/d/km$^2$)",
        "pommtMean":"Period of Mean Temperature (days)",
        "gord":"Strahler Stream Order",
        "dompMean":"Day of Mean Precipitation (days)",
        "cls7":"Proportion Cultivated and Managed Vegetation",
        "meanPercentDC_Imperfectly":"% Imperfectly Drained Soil"
        }


def getRegressors(ldf, udf, var1, var2):
    xl = np.expand_dims(ldf[var1].to_numpy(), axis=1)
    yl = ldf[var2].to_numpy()
    xu = np.expand_dims(udf[var1].to_numpy(), axis=1)
    yu = udf[var2].to_numpy()
    
    reg1 = TheilSenRegressor().fit(xl, yl)
    reg2 = TheilSenRegressor().fit(xu, yu)
    
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
    axs[1].set_xlabel("absolute difference in slope given\na random selection of catchments", fontsize=17) 
    axs[1].set_ylabel("count", fontsize=17)
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

def plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=0, upperBound=1, cmap="seismic"):

    sortIndices = np.argsort(df[var3])
    #colors = getColors
    #m = getM(variable=var3, cmap="seismic", df=df)
    #colors = getColors(var3, m, df)
    
    norm = plt.Normalize(vmin=lowerBound, vmax=upperBound)
    fig, ax = plt.subplots(figsize=(8,8)) #nrows=1, ncols=1, figsize=(16,4))
    scatter = ax.scatter(x=np.array(df[var1])[sortIndices], y=np.array(df[var2])[sortIndices], c=np.array(df[var3])[sortIndices], cmap=cmap, norm=norm)
    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical")

    cbar.set_label(varToTitle[var3], fontsize=17)
    cbar.ax.tick_params(labelsize=17)
    legend = plt.legend()
    for label in legend.get_texts():
        label.set_fontsize(17)  # Set the desired fontsize (e.g., 12)

        # Adjust the marker size for legend handles (icons)
        for handle in legend.legendHandles:
                handle.set_sizes([40])
 

    if var1 in predictablesToPretty.keys():
        ax.set_xlabel(predictablesToPretty[var1], fontsize=17)
    elif var1 in predictorsToPretty.keys():
        ax.set_xlabel(predictorsToPretty[var1], fontsize=17)
    elif var1 in predictorsToPrettyPCA.keys():
        ax.set_xlabel(predictorsToPrettyPCA[var1], fontsize=17)
    else:
        ax.set_xlabel(var1)

    if var2 in predictablesToPretty.keys():
        ylabel = predictablesToPretty[var2]
    elif var2 in predictorsToPretty.keys():
        ylabel = predictorsToPretty[var2]
    elif var2 in predictorsToPrettyPCA.keys():
        ylabel = predictorsToPrettyPCA[var2]
    ax.set_ylabel(ylabel, fontsize=17)
    
    if log:
        ax.yscale("log")
        ax.xscale("log")
    
    var1Range = np.expand_dims(np.linspace(np.min(df[var1]), np.max(df[var1])), axis=1)
    lowerPreds = reg1.predict(var1Range)
    upperPreds = reg2.predict(var1Range)
    
    if cmap == "seismic": # make sure the lines match the color scale
        ax.plot(var1Range, lowerPreds, c="b", label=predictorsToPretty[var3] + " < " + str(threshold))
        ax.plot(var1Range, upperPreds, c="r", label=predictorsToPretty[var3] + ">=" + str(threshold))
    else:
        ax.plot(var1Range, lowerPreds, c="r", label=predictorsToPretty[var3] + " < " + str(threshold))
        ax.plot(var1Range, upperPreds, c="b", label=predictorsToPretty[var3] + ">=" + str(threshold))

    ax.legend()
    ax.set_ylim(np.min(df[var2]) - (np.std(df[var2]) / 3.), np.max(df[var2]) + - (np.std(df[var2]) / 3.))

    ax.grid()

    #return fig, axs
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, str(var2) + "_" + str(var1) + "_" + str(var3) + ".png"))
    plt.clf()
    plt.close()


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



def calculateSensitivities(numIterations=1e2):
    predictorsToPretty["cls3_100"] = "% deciduous broadleaf trees"#predictorsToPretty["cls3"]
    predictorsToPretty["cls5_100"] = "% shrubs" #predictorsToPretty["cls5"]

    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(dataFilePath)
    df = df[df["domfSlope"] < 30] # remove a single outlier
    df = df[~df["pommfSlope"].isna()]
    #df = df.dropna()
    df["cls3_100"]  = df["cls3"] * 100
    df["cls5_100"]  = df["cls5"] * 100

    with open(os.path.join(outputFilesPath, "sensitivities.txt"), "w+") as oFile:

        # new Figure 4 *********************************************************

        var1 = "p_petPercentChange"
        var2 = "d_pPercentChange"
        var3 = "cls7"
        threshold = .20
        lowerBound = 0
        upperBound = 1
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic")

        var1 = "p_petPercentChange"
        var2 = "d_pPercentChange"
        var3 = "m"
        threshold = 1.5
        lowerBound = 0
        upperBound = 5
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic_r")

        

        var1 = "p_petPercentChange"
        var2 = "d_pPercentChange"
        var3 = "meanPercentDC_Imperfectly"
        #plt.hist(df["meanPercentDC_Imperfectly"])
        #plt.savefig(os.path.join(figurePath, "test.png"))
        threshold = 20
        lowerBound = np.min(df["meanPercentDC_Imperfectly"]) + 5
        upperBound = np.max(df["meanPercentDC_Imperfectly"]) - 10
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic")


        var1 = "p_petPercentChange"
        var2 = "d_pPercentChange"
        var3 = "pommtMean"
        #plt.hist(df["pommtMean"])
        #plt.savefig(os.path.join(figurePath, "test.png"))
        #quit()
        threshold = 280
        lowerBound = np.min(df["pommtMean"]) + 50
        upperBound = np.max(df["pommtMean"])
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic")

        var1 = "p_petPercentChange"
        var2 = "d_pPercentChange"
        var3 = "gord"
        threshold = 5
        lowerBound = 0
        upperBound = 12
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic")

        var1 = "p_petPercentChange"
        var2 = "d_pPercentChange"
        var3 = "dompMean"
        #plt.hist(df["dompMean"])
        #plt.savefig(os.path.join(figurePath, "test.png"))
        threshold = 215
        lowerBound = np.min(df["dompMean"])
        upperBound = np.max(df["dompMean"]) + 10
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic_r")
        
        quit()
        # Figure 4 *************************************************************
         
        var1 = "pet_pPercentChange"
        var2 = "d_pPercentChange"
        var3 = "cls3_100"
        threshold = 5
        lowerBound = 0
        upperBound = 100 
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic_r")
        #pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)


        var1 = "pet_pPercentChange"
        var2 = "d_pPercentChange"
        var3 = "maspMean"
        threshold = 1.3e6
        lowerBound = 0
        upperBound = np.max(df["maspMean"])
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic_r")
        #pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
       
         
        var1 = "pet_pPercentChange"
        var2 = "d_pPercentChange"
        var3 = "m"
        threshold = 1.5
        lowerBound = 0
        upperBound = 5
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic_r")
        #pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        #writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        

        # Figure 5
        ''' 
        var1 = "maspPercentChange"
        var2 = "d_pPercentChange"
        var3 = "maspMean"
        threshold = 1.5e6
        lowerBound = 0
        upperBound = np.max(df["maspMean"])
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic_r")
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        
        
        var1 = "maspPercentChange"
        var2 = "d_pPercentChange"
        var3 = "cls3_100"
        threshold = 5
        lowerBound = 0
        upperBound = 100
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic_r")
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        
         
        var1 = "maspPercentChange"
        var2 = "d_pPercentChange"
        var3 = "m"
        threshold = 1.5
        lowerBound = 0
        upperBound = 5
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic_r")
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        '''  

        # legacy code starts below

        '''
        var1 = "maspPercentChange"
        var2 = "masdPercentChange"
        var3 = "m"
        threshold = 1.5
        lowerBound = 0
        upperBound = 5
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound)
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
         
        var1 = "p_petSlope"
        var2 = "pommfSlope"
        var3 = "meanPercentDC_ModeratelyWell"
        threshold = 0.5
        lowerBound = 0
        upperBound = 100
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound)
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        
        var1 = "dompSlope"
        var2 = "domfSlope"
        var3 = "cls5_100"
        threshold = 0.05
        lowerBound = 0
        upperBound = 1
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound)
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        
        var1 = "dompSlope"
        var2 = "domfSlope"
        var3 = "cls3_100"
        threshold = 0.05
        lowerBound = 0
        upperBound = 1
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound)
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
         
        
        var1 = "pet_pSlope"
        var2 = "d_pPercentChange"
        var3 = "m"
        threshold = 1.5
        lowerBound = 0
        upperBound = 5
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound)
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)

         
        
        var1 = "maspPercentChange"
        var2 = "d_pPercentChange"
        var3 = "maspMean"
        threshold = 1.5e6
        lowerBound = 0
        upperBound = np.max(df["maspMean"])
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic_r")
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
         
                         
        var1 = "maspPercentChange"
        var2 = "d_pPercentChange"
        var3 = "m"
        threshold = 1.5
        lowerBound = 0
        upperBound = 5
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound)
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        
        # FIXME: redo the one right below this (not statistically significant when tried with 1000)
        #var1 = "pet_pSlope"
        var1 = "pet_pPercentChange"
        var2 = "d_pPercentChange"
        var3 = "cls3_100"
        threshold = 5
        lowerBound = 0
        upperBound = 100
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound)
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
         
        #var1 = "pet_pSlope"
        var1 = "pet_pPercentChange"
        var2 = "d_pPercentChange"
        var3 = "maspMean"
        threshold = 1.5e6
        lowerBound = 0
        upperBound = np.max(df["maspMean"])
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound, cmap="seismic_r")
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
         
        var1 = "maspPercentChange"
        var2 = "d_pPercentChange"
        var3 = "cls3_100"
        threshold = 5
        lowerBound = 0
        upperBound = 100
        slopeL, slopeU, rSquaredL, rSquaredU, reg1, reg2 = getSensitivity(df, var1, var2, var3, threshold)
        fig, axs = plotVar(df, var1, var2, var3, reg1, reg2, threshold, log=False, lowerBound=lowerBound, upperBound=upperBound)
        pval = getPVal(df, var1, var2, var3, threshold, fig, axs, numIterations)
        writeSensitiity(oFile, slopeL, slopeU, rSquaredL, rSquaredU, var1, var2, var3, threshold, pval)
        '''  
    del predictorsToPretty["cls3_100"]
    del predictorsToPretty["cls5_100"]
