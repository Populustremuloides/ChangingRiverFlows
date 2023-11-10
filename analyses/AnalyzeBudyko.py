import os
import pandas as pd
from data.metadata import *
from analyses.colorCatchments import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.stats import spearmanr

def analyzeBudyko():
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(dataFilePath)
    colorVar = "d_pPercentChange"
    cmap = "seismic"
    xVar = "p_petMean"
    yVar = "d_pMean"

    ## estimate m
    p_pets = np.linspace(0, 4.5, 100)
    ws = [1,90]
    for w in ws:
        ys = np.power(1 + np.power(p_pets, -1 * w), 1. / w)  - np.power(p_pets, -1)
        plt.plot(p_pets, ys, label="m=" + str(w))

    m = getM(colorVar, cmap, df)
    cs = getColors(colorVar, m, df, transform=None)
    plt.scatter(x=df[xVar], y=df[yVar], c=cs, alpha=0.5)
    plt.ylim(-0.1, 1.2)
    plt.show()
    




