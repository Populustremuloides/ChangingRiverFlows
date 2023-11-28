import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from colorCatchments import *
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
        "d_pPercentChange":"Percent Change in Runoff Ratio",
        "m":"Fuh's Parameter",
        "budget_deficit":"Budget Deficit"
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


def plotVar(var, m, df, tag):
    # width, height
    fig = plt.figure(figsize=(9 * 2, 6 * 1.5))

    ax = fig.add_subplot(1,1,1, projection=ccrs.InterruptedGoodeHomolosine()) #Robinson()) #ccrs.PlateCarree())
    ax.set_global()
    #ax.set_extent([-180,180,-58,83], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    
    sortingIndices = np.argsort(df[var])

    colors = getColors(var, m, df)
    plt.scatter(x=np.array(df["Longitude"])[sortingIndices], y=np.array(df["Latitude"])[sortingIndices], c=np.array(colors)[sortingIndices], s=5, alpha=0.9, transform=ccrs.PlateCarree())
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, tag + "_" + varToTitleS[var]), dpi=300)
    plt.clf()
    plt.close()


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
    
    m = getM_budget_deficit(cmap, df)
    plotVar("budget_deficit", m, df, tag)
    colorbar_budget_deficit(cmap, df)
    colorbar_budget_deficit(cmap, df, pLeft=True)
    loop.update(1)

    # Fuh's Parameter

    m = getM_M(cmap, df)
    plotVar("m", m, df, tag)
    colorbar_M(cmap, df)
    colorbar_M(cmap, df, pLeft=True)
    loop.update(1)

    # runoff ratio

    m = getM_d_pMean(cmap_r, df)
    plotVar("d_pMean", m, df, tag)
    colorbar_d_pMean(cmap_r, df)
    colorbar_d_pMean(cmap_r, df, pLeft=True)
    loop.update(1)

    m = getM_d_pSlope(cmap_r, df)
    plotVar("d_pSlope", m, df, tag)
    colorbar_d_pSlope(cmap_r, df)
    colorbar_d_pSlope(cmap_r, df, pLeft=True)
    loop.update(1)

    m = getM_d_pPercentChange(cmap_r, df)
    plotVar("d_pPercentChange", m, df, tag)
    colorbar_d_pPercentChange(cmap_r, df)
    colorbar_d_pPercentChange(cmap_r, df, pLeft=True)
    loop.update(1)

    # mean annual specific discharge

    m = getM_masdMean(cmap_r, df)
    plotVar("masdMean", m, df, tag)
    colorbar_masdMean(cmap_r, df)
    colorbar_masdMean(cmap_r, df, pLeft=True)
    loop.update(1)

    m = getM_masdSlope(cmap_r, df)
    plotVar("masdSlope", m, df, tag)
    colorbar_masdSlope(cmap_r, df)
    colorbar_masdSlope(cmap_r, df, pLeft=True)
    loop.update(1)

    m = getM_masdPercentChange(cmap_r, df)
    plotVar("masdPercentChange", m, df, tag)
    colorbar_masdPercentChange(cmap_r, df)
    colorbar_masdPercentChange(cmap_r, df, pLeft=True)
    loop.update(1)

    # period of mean flow

    m = getM_pommfMean(cmap, df)
    plotVar("pommfMean", m, df, tag)
    colorbar_pommfMean(cmap, df)
    colorbar_pommfMean(cmap, df, pLeft=True)
    loop.update(1)

    m = getM_pommfSlope(cmap, df)
    plotVar("pommfSlope", m, df, tag)
    colorbar_pommfSlope(cmap, df)
    colorbar_pommfSlope(cmap, df, pLeft=True)
    loop.update(1)

    # day of mean flow

    m = getM_domfMean(cmap, df)
    plotVar("domfMean", m, df, tag)
    colorbar_domfMean(cmap, df)
    colorbar_domfMean(cmap, df, pLeft=True)
    loop.update(1)

    m = getM_domfSlope(cmap, df)
    plotVar("domfSlope", m, df, tag)
    colorbar_domfSlope(cmap, df)
    colorbar_domfSlope(cmap, df, pLeft=True)
    loop.update(1)

    # day of peak flow

    m = getM_dopfMean(cmap, df)
    plotVar("dopfMean", m, df, tag)
    colorbar_dopfMean(cmap, df)
    colorbar_dopfMean(cmap, df, pLeft=True)
    loop.update(1)

    m = getM_dopfSlope(cmap, df)
    plotVar("dopfSlope", m, df, tag)
    colorbar_dopfSlope(cmap, df)
    colorbar_dopfSlope(cmap, df, pLeft=True)
    loop.update(1)

