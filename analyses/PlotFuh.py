import os
import pandas as pd
from data.metadata import *
from analyses.colorCatchments import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from colorCatchments import getColors
from scipy.stats import spearmanr
import copy
from sklearn.linear_model import TheilSenRegressor

def plotFuh():
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    df = pd.read_csv(dataFilePath)
    df = df[~df["d_pSlope"].isna()]

    colorVar = "d_pSlope"
    xVar = "p_petMean"
    yVar = "d_pMean"

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13, 4.5))
    p_pets = np.linspace(0, np.max(df["p_petMean"]) + 0.5, 100)
    for w in [1, 90]:
        ys = np.power(1 + np.power(p_pets, -1 * w), 1. / w)  - np.power(p_pets, -1)
        if w == 1:
            axs[0].plot(p_pets, ys, c="k", linestyle="--", label="m=1")
        else:
            axs[0].plot(p_pets, ys, c="k", linestyle="-", label="m=$\infty$")

    #m = getM(colorVar, cmap, df)
    #cs = getColors(colorVar, m, df, transform=None)
    

    lowerBound = 0
    upperBound = 8
    norm = plt.Normalize(vmin=lowerBound, vmax=upperBound) 
    scatterPlot = axs[0].scatter(x=df[xVar], y=df[yVar], c=df["m"], cmap="seismic", norm=norm, alpha=0.7)
    axs[0].set_ylim(-0.1, 1.2)
    axs[0].set_xlabel(predictorsToPretty[xVar])
    axs[0].set_ylabel(predictorsToPretty[yVar])
    axs[0].legend()
    axs[0].set_title("Fuh's Equation")
    axs[0].grid()

    #fig, ax = plt.subplots(figsize=(9,4.4))   
    # create a histogram using those outputs
    N, bins, patches = axs[1].hist(df["m"], bins=50, density=False)
    maxHeight = np.max(N)
    cmap = scatterPlot.get_cmap()

    # color the various sections
    for i, bini in enumerate(bins[1:]):
        patches[i].set_facecolor(cmap(norm(bini)))

    axs[1].set_title("Distribution of Fuh's Parameter")
    axs[1].set_xlabel("m value")
    axs[1].set_ylabel("count")
    axs[1].grid()   

    cAreas = np.log10(np.array(df["Catchment Area"]) + 1)
    norm = plt.Normalize(vmin=np.min(cAreas), vmax=np.max(cAreas))
    scatter = axs[2].scatter(df["forest"], df["m"], c=cAreas, cmap="PuOr", norm=norm)
    cbar = fig.colorbar(scatter, ax=axs[2], orientation="vertical")
    cbar.set_label("Log$_{10}$ Catchment Area (km$^2$)", fontsize=10)
    cbar.ax.tick_params(labelsize=12)
    axs[2].set_xlabel("Proportion Forest Cover")
    axs[2].set_ylabel("m value")
    axs[2].grid()   
    axs[2].set_title("Correlates of Fuh's Paramter")
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "fuh_forest_and_area_oh_my.png"))
    plt.clf()
    plt.close()

    # plot Fuh's Parameter's relationship with the water budget deficit 

    fig = plt.figure(figsize=(13, 4.5))
    axs = []
    axs.append(fig.add_subplot(131))
    axs.append(fig.add_subplot(132))
    axs.append(fig.add_subplot(133, sharey=axs[-1], sharex=axs[-1]))
    #fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(13, 4.5))
    
    norm = plt.Normalize(vmin=0, vmax=8)
    colorVar = "m"
    sortIndices = np.argsort(df[colorVar])
    scatter = axs[0].scatter(np.array(df["forest"])[sortIndices], np.array(df["percent_deficit"])[sortIndices], c=np.array(df[colorVar])[sortIndices], norm=norm, cmap="seismic")
    cbar = fig.colorbar(scatter, ax=axs[0], orientation="vertical")
    cbar.set_label("Fuh's Paramter", fontsize=10)
    cbar.ax.tick_params(labelsize=12)
    axs[0].set_title("$\\frac{P - ET - Q}{P}$ (% Water Budget Deicit)")
    axs[0].set_xlabel("Forest Cover")
    axs[0].set_ylabel("% deficit", fontsize=12)
    #axs[0].set_xscale("log")
    axs[0].grid()

    scatter = axs[1].scatter(np.array(df["Catchment Area"]), np.array(df["maspMean"]) - np.array(df["masetMean"]),c=df["forest"], cmap="PiYG", label="P - ET") 
    #cbar = fig.colorbar(scatter, ax=axs[1], orientation="vertical")
    #cbar.set_label("Forest Cover", fontsize=10)
    #cbar.ax.tick_params(labelsize=12)
    axs[1].set_title("P - ET (potential discharge)")
    axs[1].set_xlabel("Catchment Area")
    axs[1].set_ylabel("Liters per Square Kilometer / Year", fontsize=12)
    axs[1].set_xscale("log")
    axs[1].set_ylim(-4e6, 8e6)
    axs[1].grid()

    scatter = axs[2].scatter(np.array(df["Catchment Area"]), np.array(df["masdMean"]), c=df["forest"], cmap="PiYG", label="P - ET") 
    cbar = fig.colorbar(scatter, ax=axs[2], orientation="vertical")
    cbar.set_label("Forest Cover", fontsize=10)
    cbar.ax.tick_params(labelsize=12)
    axs[2].set_title("Q (discharge)")
    axs[2].set_xlabel("Catchment Area")
    axs[2].set_ylabel("Liters per Square Kilometer / Year", fontsize=12)
    axs[2].set_xscale("log")
    axs[2].grid()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "fuh_and_budget_deficit.png"))
    plt.clf()
    plt.close()
    

