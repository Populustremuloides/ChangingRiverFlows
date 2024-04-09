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


mpl.use('Agg')

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
        "masdMean":"Mean Annual Specific Dicharge\n(L/d/km$^2$)",
        "masdSlope":"Change in Mean Annual\nSpecific Discharge (L/d/km$^2$) / year",
        "masdPercentChange":"Percent Change in Mean\nAnnual Specific Discharge",
        "domfMean":"Day of Mean Flow",
        "domfSlope":"Change in Day of Mean Flow\n(days / year)",
        "dopfMean":"Day of Peak Flow (days)",
        "dopfSlope":"Change in Day of Peak Flow (days /year)",
        "pommfMean":"Period of Mean Flow (days)",
        "pommfSlope":"Change in Period of Mean Flow\n(days / year)",
        "d_pMean":"Runoff Ratio",
        "d_pSlope":"Change in Runoff Ratio per Year",
        "d_pPercentChange":"Percent Change in\nRunoff Ratio per Year",
        "m":"Fuh's Parameter (m)",
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


def plotVar(var, df, dfAll, ax, fig, lowerBound, upperBound, logFile, cmap="seismic", randomForest=False):
    # width, height

    # Remove the tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Remove the tick marks
    ax.tick_params(axis='both', which='both', length=0)

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
    cbar.set_label(varToTitle[var], fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    cbar.ax.yaxis.get_offset_text().set_fontsize(25)

    legend = plt.legend()
    for label in legend.get_texts():
        label.set_fontsize(23)  # Set the desired fontsize (e.g., 12)

        # Adjust the marker size for legend handles (icons)
        for handle in legend.legendHandles:
                handle.set_sizes([70])
    plt.tight_layout()

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
        #plt.grid(False)
    

        # Deficit Figure ***************************************************************
        fig = plt.figure(figsize=(9 * 2, 6 * 1.5))
        #fig, ax = plt.subplots() 
        #fig.patch.set_edgecolor('none')
        #fig.patch.set_linewidth(0)



        ax = fig.add_subplot(1,1,1, projection=ccrs.InterruptedGoodeHomolosine())

        # Remove the tick labels
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        # Remove the tick marks
        #ax.tick_params(axis='both', which='both', length=0)

        lowerBound = -250 #np.min(np.sort(df["budget_deficit"])[50:])
        upperBound = 250 #-1 * lowerBound
        plotVar("percent_deficit", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        plt.savefig(os.path.join(figurePath, varToTitleS["percent_deficit"] + "_GD.png"), dpi=300)        
        plt.clf()
        plt.close()

        # Fuh Figure *******************************************************************
        fig = plt.figure(figsize=(9 * 2, 6 * 1.5)) 
        ax = fig.add_subplot(1,1,1, projection=ccrs.InterruptedGoodeHomolosine())

        lowerBound = 0
        upperBound = 8
        plotVar("m", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, randomForest=randomForest)
        loop.update(1)

        plt.savefig(os.path.join(figurePath, varToTitleS["m"] + "_GD.png"), dpi=300,  x_inches='tight', pad_inches=0, frameon=False)
        plt.clf()
        plt.close()
        
        # Main Text Figure ***********************************************************
        
        fig = plt.figure(figsize=(16 * 2.3, 13 * 2.3))

        # runoff ratio
        ax = fig.add_subplot(4, 2, 1, projection=ccrs.InterruptedGoodeHomolosine())
        lowerBound = -10
        upperBound = 10
        plotVar("d_pPercentChange", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        ax = fig.add_subplot(4, 2, 2, projection=ccrs.InterruptedGoodeHomolosine())
        lowerBound = 0
        upperBound = 1
        plotVar("d_pMean", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        # mean annual specific discharge
        ax = fig.add_subplot(4, 2, 3, projection=ccrs.InterruptedGoodeHomolosine())
        lowerBound = -10
        upperBound = 12
        plotVar("masdPercentChange", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        ax = fig.add_subplot(4, 2, 4, projection=ccrs.InterruptedGoodeHomolosine())
        lowerBound = 0
        upperBound = 5e6
        plotVar("masdMean", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        # day of mean flow
        ax = fig.add_subplot(4, 2, 5, projection=ccrs.InterruptedGoodeHomolosine())
        lowerBound = -9
        upperBound = 9
        plotVar("domfSlope", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, randomForest=randomForest)
        loop.update(1)

        ax = fig.add_subplot(4, 2, 6, projection=ccrs.InterruptedGoodeHomolosine())
        lowerBound = 110
        upperBound = 260
        plotVar("domfMean", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, randomForest=randomForest)
        loop.update(1)

        # period of mean flow
        ax = fig.add_subplot(4, 2, 7, projection=ccrs.InterruptedGoodeHomolosine())
        lowerBound = -10
        upperBound = 10
        plotVar("pommfSlope", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        ax = fig.add_subplot(4, 2, 8, projection=ccrs.InterruptedGoodeHomolosine())
        lowerBound = 20
        upperBound = 300
        plotVar("pommfMean", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        if randomForest:
            plt.savefig(os.path.join(figurePath, "map_mainText_RF.png"), dpi=300, x_inches='tight', pad_inches=0, frameon=False)
        else:
            plt.savefig(os.path.join(figurePath, "map_mainText_GD.png"), dpi=300, x_inches='tight', pad_inches=0, frameon=False)
        plt.clf()
        plt.close()

        # Supplemental Figure *******************************************************************************

        fig = plt.figure(figsize=(18 * 2, 13 * 2))

        # Runoff Ratio (raw)        

        lowerBound = -0.03
        upperBound = 0.03
        ax = fig.add_subplot(3, 2, 1, projection=ccrs.InterruptedGoodeHomolosine())
        plotVar("d_pSlope", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        lowerBound = 0
        upperBound = 1
        ax = fig.add_subplot(3, 2, 2, projection=ccrs.InterruptedGoodeHomolosine())
        plotVar("d_pMean", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        # mean annual specific discharge (raw)
        
        lowerBound = -1e5
        upperBound = 1e5
        ax = fig.add_subplot(3, 2, 3, projection=ccrs.InterruptedGoodeHomolosine())
        plotVar("masdSlope", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        lowerBound = 0
        upperBound = 5e6
        ax = fig.add_subplot(3, 2, 4, projection=ccrs.InterruptedGoodeHomolosine())
        plotVar("masdMean", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r", randomForest=randomForest)
        loop.update(1)

        # day of peak flow

        lowerBound = -20
        upperBound = 20
        ax = fig.add_subplot(3, 2, 5, projection=ccrs.InterruptedGoodeHomolosine())
        plotVar("dopfSlope", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, randomForest=randomForest)
        loop.update(1)

        lowerBound = 100
        upperBound = 300
        ax = fig.add_subplot(3, 2, 6, projection=ccrs.InterruptedGoodeHomolosine())
        plotVar("dopfMean", df, dfAll, ax, fig, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, randomForest=randomForest)
        loop.update(1)

        if randomForest:
            plt.savefig(os.path.join(figurePath, "map_supplemental_RF.png"), dpi=300)
        else:
            plt.savefig(os.path.join(figurePath, "map_supplemental_GD.png"), dpi=300)
        plt.clf()
        plt.close()
