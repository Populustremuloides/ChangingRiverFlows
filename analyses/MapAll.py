import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from colorCatchments import *
from tqdm import tqdm
df = pd.read_csv(os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata.csv"))


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
        "masdMean":"Mean Annual Specific Dicharge",
        "masdSlope":"Change in Mean Annual Specific Discharge",
        "masdPercentChange":"Percent Change in Mean Annual Specific Discharge",
        "domfMean":"Day of Mean Flow",
        "domfSlope":"Change in Day of Mean Flow",
        "dopfMean":"Day of Peak Flow",
        "dopfSlope":"Change in Day of Peak Flow",
        "pommfMean":"Period of Mean Flow",
        "pommfSlope":"Change in Period of Mean Flow",
        "d_pMean":"Runoff Ratio",
        "d_pSlope":"Change in Runoff Ratio",
        "d_pPercentChange":"Percent Change in Runoff Ratio"
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
        "d_pPercentChange":"MAP_RunoffRatioPercentChange.png"
        }


def plotVar(var, m):
    # width, height
    fig = plt.figure(figsize=(11 * 2, 6 * 1.5))

    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    ax.set_extent([-180,180,-58,83], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)

    colors = getColors(var, m, df)
    plt.scatter(x=df["Longitude"], y=df["Latitude"], c=colors, s=5, alpha=0.9)

    plt.savefig(os.path.join(figurePath, varToTitleS[var]), dpi=300)
    plt.clf()
    plt.close()


def mapAll():

    loop = tqdm(total=12)
    loop.set_description("mapping catchments")
    cmap = seismic_r

    # runoff ratio

    m = getM_d_pMean(cmap)
    plotVar("d_pMean", m)
    colorbar_d_pMean(cmap)
    colorbar_d_pMean(cmap, pLeft=True)
    loop.update(1)

    m = getM_d_pSlope(cmap)
    plotVar("d_pSlope", m)
    colorbar_d_pSlope(cmap)
    colorbar_d_pSlope(cmap, pLeft=True)
    loop.update(1)

    m = getM_d_pPercentChange(cmap)
    plotVar("d_pPercentChange", m)
    colorbar_d_pPercentChange(cmap)
    colorbar_d_pPercentChange(cmap, pLeft=True)
    loop.update(1)

    # mean annual specific discharge

    m = getM_masdMean(cmap)
    plotVar("masdMean", m)
    colorbar_masdMean(cmap)
    colorbar_masdMean(cmap, pLeft=True)
    loop.update(1)

    m = getM_masdSlope(cmap)
    plotVar("masdSlope", m)
    colorbar_masdSlope(cmap)
    colorbar_masdSlope(cmap, pLeft=True)
    loop.update(1)

    m = getM_masdPercentChange(cmap)
    plotVar("masdPercentChange", m)
    colorbar_masdPercentChange(cmap)
    colorbar_masdPercentChange(cmap, pLeft=True)
    loop.update(1)

    # period of mean flow

    m = getM_pommfMean(cmap)
    plotVar("pommfMean", m)
    colorbar_pommfMean(cmap)
    colorbar_pommfMean(cmap, pLeft=True)
    loop.update(1)

    m = getM_pommfSlope(cmap)
    plotVar("pommfSlope", m)
    colorbar_pommfSlope(cmap)
    colorbar_pommfSlope(cmap, pLeft=True)
    loop.update(1)

    # day of mean flow

    m = getM_domfMean(cmap)
    plotVar("domfMean", m)
    colorbar_domfMean(cmap)
    colorbar_domfMean(cmap, pLeft=True)
    loop.update(1)

    m = getM_domfSlope(cmap)
    plotVar("domfSlope", m)
    colorbar_domfSlope(cmap)
    colorbar_domfSlope(cmap, pLeft=True)
    loop.update(1)

    # day of peak flow

    m = getM_dopfMean(cmap)
    plotVar("dopfMean", m)
    colorbar_dopfMean(cmap)
    colorbar_dopfMean(cmap, pLeft=True)
    loop.update(1)

    m = getM_dopfSlope(cmap)
    plotVar("dopfSlope", m)
    colorbar_dopfSlope(cmap)
    colorbar_dopfSlope(cmap, pLeft=True)
    loop.update(1)

