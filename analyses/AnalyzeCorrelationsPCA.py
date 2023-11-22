import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# cycle through the data and calculate the runoff ratios

def _analzye(df):
    # some minor cleaning of up variables that aren't numeric

    df = df.drop(["catchment"], axis=1)#,"quality","River","Station","Country","LINKNO","Ecoregion_Name","Continent","BIOME","ECO_NAME"], axis=1)

    seismic = mpl.cm.get_cmap('seismic')

    # exploratory analysis of how changes in flow are related to each other
    corr = df.corr()
    sns.clustermap(corr, annot=False, xticklabels=1, yticklabels=1, figsize=(14,13), cmap=seismic, center=0)
    correlationFigureOfPCAPath = os.path.join(figurePath, "clustermapOfAll_PCA.png")
    plt.savefig(correlationFigureOfPCAPath)
    plt.clf()

def analyzeCorrelationsPCA():
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedPCA.csv")
    df = pd.read_csv(dataFilePath)
    df = df[np.array(~df["1"].isna())] # keep only the rows for which we have data
    _analzye(df)

