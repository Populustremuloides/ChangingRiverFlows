import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from data.metadata import *
#from ColorCatchments2 import *
#from colorCatchments import *
from tqdm import tqdm
import os

seismic = mpl.cm.get_cmap('seismic')
seismic_r = mpl.cm.get_cmap('seismic_r')
PiYG = mpl.cm.get_cmap('PiYG')
PiYG_r = mpl.cm.get_cmap('PiYG_r')
cool = mpl.cm.get_cmap('cool')
cool_r = mpl.cm.get_cmap('cool_r')
plasma = mpl.cm.get_cmap('plasma')
plasma_r = mpl.cm.get_cmap('plasma_r')
viridis = mpl.cm.get_cmap('viridis')
viridis_r = mpl.cm.get_cmap('viridis_r')
PuOr = mpl.cm.get_cmap("PuOr")
PuOr_r = mpl.cm.get_cmap("PuOr_r")

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
        "percent_deficit":"% Budget Deficit"
        }



varToTitleS = {
        "masdMean":"MAP_MeanAnnualSpecificDicharge",
        "masdSlope":"MAP_ChangeinMeanAnnualSpecificDischarge",
        "masdPercentChange":"MAP_PercentChangeinMeanAnnualSpecificDischarge",
        "domfMean":"MAP_DayofMeanFlow",
        "domfSlope":"MAP_ChangeinDayofMeanFlow",
        "dopfMean":"MAP_DayofPeakFlow",
        "dopfSlope":"MAP_ChangeinDayofPeakFlow",
        "pommfMean":"MAP_PeriodOfMeanFlowMean",
        "pommfSlope":"MAP_PeriodOfMeanFlowSlope",
        "d_pMean":"MAP_RunoffRatioMean",
        "d_pSlope":"MAP_RunoffRatioSlope",
        "d_pPercentChange":"MAP_RunoffRatioPercentChange",
        "m":"MAP_FuhsParameter",
        "budget_deficit":"MAP_budget_deficit",
        "percent_deficit":"MAP_percent_deficit"
        }


def plotVar(var, df, dfAll, lowerBound, upperBound, logFile, cmap="seismic", randomForest=False):
    # width, height
    fig = plt.figure(figsize=(9 * 2, 6 * 1.5))

    ax = fig.add_subplot(1,1,1, projection=ccrs.InterruptedGoodeHomolosine()) #Robinson()) #ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE)
    sortingIndices = np.argsort(df[var])
    sortingIndicesAll = np.argsort(dfAll[var])

    # truncate the data for visualization purposes
    norm = plt.Normalize(vmin=lowerBound, vmax=upperBound)
    #percentTruncated = 100. * ((np.sum(maxMask) + np.sum(minMask)) / minMask.shape[0])
    #logFile.write(varToTitle[var] + " was " + str(percentTruncated) + " percent truncated when plotted\n")

    # plot the imputed ones
    scatter = ax.scatter(x=np.array(dfAll["Longitude"])[sortingIndicesAll], y=np.array(dfAll["Latitude"])[sortingIndicesAll], c=np.array(dfAll[var])[sortingIndicesAll], cmap=cmap, norm=norm, s=10, alpha=0.9, transform=ccrs.PlateCarree(), marker="_", label="imputed")

    # plot the real values
    scatter = ax.scatter(x=np.array(df["Longitude"])[sortingIndices], y=np.array(df["Latitude"])[sortingIndices], c=np.array(df[var])[sortingIndices], cmap=cmap, norm=norm, s=5, alpha=0.9, transform=ccrs.PlateCarree(), label="measured")
    
    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_label(varToTitle[var], fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    legend = plt.legend()
    for label in legend.get_texts():
        label.set_fontsize(16)  # Set the desired fontsize (e.g., 12)

        # Adjust the marker size for legend handles (icons)
        for handle in legend.legendHandles:
                handle.set_sizes([50])
    plt.tight_layout()
    if randomForest:
        plt.savefig(os.path.join(figurePath, varToTitleS[var] + "_RF.png"), dpi=300)
    else:
        plt.savefig(os.path.join(figurePath, varToTitleS[var] + "_GD.png"), dpi=300)
    plt.clf()
    plt.close()

def mapAll(randomForest=False):

    dfPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    if os.path.exists(dfPath):
        df = pd.read_csv(dfPath)
    
    if randomForest:
        dfPathAll = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedRF.csv")
        if os.path.exists(dfPathAll):
            dfAll = pd.read_csv(dfPathAll)
    else:
        dfPathAll = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedAll.csv")
        if os.path.exists(dfPathAll):
            dfAll = pd.read_csv(dfPathAll)


    mask = np.array(df["d_pSlope"].isna())
    df = df[~mask]    
    dfAll = dfAll[mask]

    loop = tqdm(total=12)
    loop.set_description("mapping catchments")

    # Budget Deficits
    with open(os.path.join(logPath, "log_mappAll.txt"), "w+") as logFile:

        lowerBound = -250 #np.min(np.sort(df["budget_deficit"])[50:])
        upperBound = 250 #-1 * lowerBound
        plotVar("percent_deficit", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic", randomForest=randomForest)
        loop.update(1)

        lowerBound = 0
        upperBound = 8
        plotVar("m", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, randomForest=randomForest)
        loop.update(1)

        # runoff ratio
        lowerBound = 0
        upperBound = 1
        plotVar("d_pMean", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)


        lowerBound = -0.03
        upperBound = 0.03
        plotVar("d_pSlope", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        lowerBound = -10
        upperBound = 10
        plotVar("d_pPercentChange", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)


        # mean annual specific discharge
        lowerBound = 0
        upperBound = 5e6
        plotVar("masdMean", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        lowerBound = -1e5
        upperBound = 1e5
        plotVar("masdSlope", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        lowerBound = -10
        upperBound = 12
        plotVar("masdPercentChange", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        # period of mean flow

        lowerBound = 20
        upperBound = 300
        plotVar("pommfMean", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)


        lowerBound = -10
        upperBound = 10
        plotVar("pommfSlope", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)


        # day of mean flow

        lowerBound = 110
        upperBound = 260
        plotVar("domfMean", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, randomForest=randomForest)
        loop.update(1)

        lowerBound = -9
        upperBound = 9
        plotVar("domfSlope", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, randomForest=randomForest)
        loop.update(1)

        # day of peak flow

        lowerBound = 100
        upperBound = 300
        plotVar("dopfMean", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, randomForest=randomForest)
        loop.update(1)

        lowerBound = -20
        upperBound = 20
        plotVar("dopfSlope", df, dfAll, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, randomForest=randomForest)
        loop.update(1)
