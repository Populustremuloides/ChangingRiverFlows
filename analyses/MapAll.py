import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from ColorCatchments2 import *
#from colorCatchments import *
from tqdm import tqdm


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
        "budget_deficit":"Budget Deficit (Liters)"
        }



varToTitleS = {
        "masdMean":"MAP_MeanAnnualSpecificDicharge.png",
        "masdSlope":"MAP_ChangeinMeanAnnualSpecificDischarge.png",
        "masdPercentChange":"MAP_PercentChangeinMeanAnnualSpecificDischarge.png",
        "domfMean":"MAP_DayofMeanFlow.png",
        "domfSlope":"MAP_ChangeinDayofMeanFlow.png",
        "dopfMean":"MAP_DayofPeakFlow.png",
        "dopfSlope":"MAP_ChangeinDayofPeakFlow.png",
        "pommfMean":"MAP_PeriodOfMeanFlowMean.png",
        "pommfSlope":"MAP_PeriodOfMeanFlowSlope.png",
        "d_pMean":"MAP_RunoffRatioMean.png",
        "d_pSlope":"MAP_RunoffRatioSlope.png",
        "d_pPercentChange":"MAP_RunoffRatioPercentChange.png",
        "m":"MAP_FuhsParameter.png",
        "budget_deficit":"MAP_budget_deficit.png"
        }


def plotVar(var, df, tag, lowerBound, upperBound, logFile, cmap="seismic"):
    # width, height
    fig = plt.figure(figsize=(9 * 2, 6 * 1.5))

    ax = fig.add_subplot(1,1,1, projection=ccrs.InterruptedGoodeHomolosine()) #Robinson()) #ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE)
    sortingIndices = np.argsort(df[var])

    # truncate the data for visualization purposes
    colors = np.array(df[var])
    minMask = colors < lowerBound
    colors[minMask] = lowerBound
    maxMask = colors > upperBound
    colors[maxMask] = upperBound
    percentTruncated = 100. * ((np.sum(maxMask) + np.sum(minMask)) / minMask.shape[0])
    logFile.write(varToTitle[var] + " was " + str(percentTruncated) + " percent truncated when plotted\n")

    colors = ax.scatter(x=np.array(df["Longitude"])[sortingIndices], y=np.array(df["Latitude"])[sortingIndices], c=colors[sortingIndices], cmap=cmap, s=5, alpha=0.9, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(colors, ax=ax, orientation="vertical")
    cbar.set_label(varToTitle[var], fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, tag + "_" + varToTitleS[var]), dpi=300)

def mapAll(tag):

    dfPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_" + str(tag) + ".csv")
    if os.path.exists(dfPath):
        df = pd.read_csv(dfPath)

    df = df[~df["d_pSlope"].isna()]

    loop = tqdm(total=12)
    loop.set_description("mapping catchments")
    cmap_r = seismic_r
    cmap = seismic

    # Budget Deficits
    with open(os.path.join(logPath, "log_mappAll.txt"), "w+") as logFile:

        lowerBound = np.min(np.sort(df["budget_deficit"])[50:])
        upperBound = -1 * lowerBound
        plotVar("budget_deficit", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r")
        loop.update(1)

        lowerBound = 0
        upperBound = 10
        plotVar("m", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile)
        loop.update(1)

        # runoff ratio
        lowerBound = 0
        upperBound = 1
        plotVar("d_pMean", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r")
        loop.update(1)


        lowerBound = -0.03
        upperBound = 0.03
        plotVar("d_pSlope", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r")
        loop.update(1)

        lowerBound = -10
        upperBound = 10
        plotVar("d_pPercentChange", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r")
        loop.update(1)


        # mean annual specific discharge
        lowerBound = 0
        upperBound = 5e6
        plotVar("masdMean", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r")
        loop.update(1)

        lowerBound = -1e5
        upperBound = 1e5
        plotVar("masdSlope", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r")
        loop.update(1)

        lowerBound = -10
        upperBound = 12
        plotVar("masdPercentChange", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r")
        loop.update(1)

        # period of mean flow

        lowerBound = 20
        upperBound = 300
        plotVar("pommfMean", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r")
        loop.update(1)


        lowerBound = -10
        upperBound = 10
        plotVar("pommfSlope", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile, cmap="seismic_r")
        loop.update(1)


        # day of mean flow

        lowerBound = 110
        upperBound = 260
        plotVar("domfMean", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile)
        loop.update(1)

        lowerBound = -9
        upperBound = 9
        plotVar("domfSlope", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile)
        loop.update(1)

        # day of peak flow

        lowerBound = 100
        upperBound = 300
        plotVar("dopfMean", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile)
        loop.update(1)

        lowerBound = -20
        upperBound = 20
        plotVar("dopfSlope", df, tag, lowerBound=lowerBound, upperBound=upperBound, logFile=logFile)
        loop.update(1)
