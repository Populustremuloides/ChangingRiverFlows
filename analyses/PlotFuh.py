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
    df = df[df["percent_deficit"] > -200] # remove  an outlier

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
    
    sortIndices = df.index[np.argsort(df["m"])]

    lowerBound = 0
    upperBound = 5
    norm = plt.Normalize(vmin=lowerBound, vmax=upperBound) 
    scatterPlot = axs[0].scatter(x=df[xVar][sortIndices], y=df[yVar][sortIndices], c=df["m"][sortIndices], cmap="seismic_r", norm=norm, alpha=0.7)
    axs[0].set_ylim(-0.1, 1.2) 
    axs[0].set_xlabel("Mean value of\nP / PET", fontsize=15)
    axs[0].set_ylabel(predictorsToPretty[yVar], fontsize=15)
    axs[0].legend()
    axs[0].set_title("Fuh's Equation", fontsize=17)
    axs[0].grid()

    #fig, ax = plt.subplots(figsize=(9,4.4))   
    # create a histogram using those outputs
    N, bins, patches = axs[1].hist(df["m"], bins=50, density=False)
    maxHeight = np.max(N)
    cmap = scatterPlot.get_cmap()

    # color the various sections
    for i, bini in enumerate(bins[1:]):
        patches[i].set_facecolor(cmap(norm(bini)))

    axs[1].set_title("Distribution of Fuh's Parameter", fontsize=17)
    axs[1].set_xlabel("m value", fontsize=15)
    axs[1].set_ylabel("count", fontsize=15)
    axs[1].grid()   

    #cAreas = df["m"] #np.log10(np.array(df["Catchment Area"]) + 1)
    #norm = plt.Normalize(vmin=np.min(cAreas), vmax=np.max(cAreas))
    scatter = axs[2].scatter(df["forest"][sortIndices], df["Catchment Area"][sortIndices], c=df["m"][sortIndices], cmap="seismic_r", norm=norm)
    #cbar = fig.colorbar(scatter, ax=axs[2], orientation="vertical")
    #cbar.set_label("Log$_{10}$ Catchment Area (km$^2$)", fontsize=15)
    #cbar.ax.tick_params(labelsize=15)
    axs[2].set_yscale("log")
    axs[2].set_xlabel("Proportion Forest Cover", fontsize=15)
    axs[2].set_ylabel("Catchment Area", fontsize=15)
    axs[2].grid()   
    axs[2].set_title("Correlates of Fuh's Paramter", fontsize=17)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "fuh_forest_and_area_oh_my.png"))
    plt.clf()
    plt.close()

    # plot Fuh's Parameter's relationship with the water budget deficit 

    fig = plt.figure(figsize=(13, 4.5))
    axs = []
    axs.append(fig.add_subplot(131))
    axs.append(fig.add_subplot(132, sharey=axs[-1]))
    axs.append(fig.add_subplot(133, sharey=axs[-1]))

    norm = plt.Normalize(vmin=0, vmax=5)
    colorVar = "m"
    sortIndices = np.argsort(df[colorVar])

    # First subplot
    scatter = axs[0].scatter(np.array(df["p_petMean"])[sortIndices], np.array(df["percent_deficit"])[sortIndices], c=np.array(df[colorVar])[sortIndices], norm=norm, cmap="seismic_r")
    axs[0].set_xlabel("P / PET (wetness index)", fontsize=15)
    axs[0].set_ylabel("$\\frac{P - ET - Q}{P}$ (% Water Budget Imbalance per Year)", fontsize=12)
    axs[0].grid()

    # Second subplot
    scatter = axs[1].scatter(np.array(df["maspMean"])[sortIndices], np.array(df["percent_deficit"])[sortIndices], c=np.array(df[colorVar])[sortIndices], norm=norm, cmap="seismic_r")
    axs[1].set_xlabel("Mean Annual Specific Precipitation", fontsize=15)
    #axs[1].set_ylabel("$\\frac{P - ET - Q}{P}$ (% Water Budget Deficit)", fontsize=15)
    axs[1].set_xscale("log")
    axs[1].grid()

    # Third subplot
    scatter = axs[2].scatter(np.array(df["forest"])[sortIndices], np.array(df["percent_deficit"])[sortIndices], c=np.array(df[colorVar])[sortIndices], norm=norm, cmap="seismic_r")
    cbar = fig.colorbar(scatter, ax=axs[2], orientation="vertical")
    cbar.set_label("Fuh's Parameter", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    axs[2].set_xlabel("Proportion Forest Cover", fontsize=15)
    #axs[2].set_ylabel("$\\frac{P - ET - Q}{P}$ (% Water Budget Deficit)", fontsize=15)
    axs[2].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "fuh_and_budget_deficit.png"))
    plt.clf()
    plt.close()
