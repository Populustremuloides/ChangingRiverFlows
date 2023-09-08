import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# cycle through the data and calculate the runoff ratios
def analyzeDistributions():
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata.csv")
    df = pd.read_csv(dataFilePath)


    plt.hist("")

    # some minor cleaning of up variables that aren't numeric
    df = df.drop(["catchment","River","Station","Country","LINKNO","Ecoregion_Name","Continent","BIOME","ECO_NAME"], axis=1)

    seismic = mpl.cm.get_cmap('seismic')

    # exploratory analysis of how changes in flow are related to each other
    changesDf = df[df.columns[110:]]
    corrChanges = changesDf.corr()
    #plt.figure()
    sns.clustermap(corrChanges, annot=False, square=True, xticklabels=1, yticklabels=1, figsize=(14,13), cmap=seismic, center=0)
    correlationFigureOfChangesPath = os.path.join(figurePath, "clustermapOfChangesInFlow.png")
    plt.savefig(correlationFigureOfChangesPath)
    plt.show()

    # exploratory analysis of how predictor variables for changes in flow are related to each other
    predictorsDf = df[df.columns[:110]]
    corrPredictors = predictorsDf.corr()
    #plt.figure()
    sns.clustermap(corrPredictors, annot=False, square=True, xticklabels=1, yticklabels=1, figsize=(20,20), cmap=seismic, center=0)
    correlationFigureOfPredictorsPath = os.path.join(figurePath, "clustermapOfPredictorsForChangesInFlow.png")
    plt.savefig(correlationFigureOfPredictorsPath)
    plt.show()




#def analyzeDistributions():
    #dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata.csv")
    #df = pd.read_csv(dataFilePath)


