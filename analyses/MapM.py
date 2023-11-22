import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from colorCatchments import *
from tqdm import tqdm


seismic = mpl.cm.get_cmap('seismic')
varToTitle = {
        "m":"Fuh's Parameter"}

varToTitleS = {
        "m":"MAP_FuhsParameter.png"
        }


def plotVar(var, m, df):
    # width, height
    fig = plt.figure(figsize=(9 * 2, 6 * 1.5))

    ax = fig.add_subplot(1,1,1, projection=ccrs.InterruptedGoodeHomolosine())
    #ax.set_extent([-180,180,-58,83], crs=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE)

    colors = getColors(var, m, df)
    plt.scatter(x=df["Longitude"], y=df["Latitude"], c=colors, s=5, alpha=0.9, transform=ccrs.PlateCarree())
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, varToTitleS[var]), dpi=300)
    plt.clf()
    plt.close()


def mapM(tag):

    dfPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_" + str(tag) + ".csv")
    if os.path.exists(dfPath):
        df = pd.read_csv(dfPath)
    


    loop = tqdm(total=1)
    loop.set_description("mapping m")
    cmap = seismic_r
    
    # Fuh's Parameter (called "m" in the literature and "w" in the code to avoid overiding m which means something to matpltolib)

    m = getM_M(cmap, df)
    plotVar("m", m, df)
    colorbar_M(cmap, df)
    colorbar_M(cmap, df, pLeft=True)
    loop.update(1)

