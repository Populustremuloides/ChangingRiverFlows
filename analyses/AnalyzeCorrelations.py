import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# cycle through the data and calculate the runoff ratios

def _analzye(df, tag):
    # some minor cleaning of up variables that aren't numeric
    df = df.drop(["catchment","quality","River","Station","Country","LINKNO","Ecoregion_Name","Continent","BIOME","ECO_NAME"], axis=1)
    df = df[np.array(~df["p_petMean"].isna())]

    seismic = mpl.cm.get_cmap('seismic')

    # exploratory analysis of how changes in flow are related to each other
    changesDf = df[df.columns[111:]]

    corrChanges = changesDf.corr()

    sns.clustermap(corrChanges, annot=False, square=True, xticklabels=1, yticklabels=1, figsize=(14,13), cmap=seismic, center=0)
    correlationFigureOfChangesPath = os.path.join(figurePath, "clustermapOfChangesInFlow_" + str(tag) + ".png")
    plt.savefig(correlationFigureOfChangesPath)
    plt.clf()

    # exploratory analysis of how predictor variables for changes in flow are related to each other
    predictorsDf = df[df.columns[:111]]

    # (this should only activate with a small number of data points)
    # (it drops columns that are all zero and that will mess with the correlation calculation)
    for col in predictorsDf.columns:
        if np.nansum(predictorsDf[col]) == 0:
            predictorsDf = predictorsDf.drop(col, axis=1)
    corrPredictors = predictorsDf.corr()

    sns.clustermap(corrPredictors, annot=False, square=True, xticklabels=1, yticklabels=1, figsize=(20,20), cmap=seismic, center=0)
    correlationFigureOfPredictorsPath = os.path.join(figurePath, "clustermapOfPredictorsForChangesInFlow_" + str(tag) + ".png")
    plt.savefig(correlationFigureOfPredictorsPath)
    plt.clf()



def analyzeCorrelations():
    tags = ["raw","imputed"]

    for tag in tags:
        dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_" + str(tag) + ".csv")
        df = pd.read_csv(dataFilePath)
        _analzye(df, tag)


