import os
from data.metadata import *
import pandas as pd

def _combine(df, tag):

    for file in os.listdir(outputFilesPath):
        if file.startswith("timeseriesSummary"):

            tdf = pd.read_csv(os.path.join(outputFilesPath, file))
            df = df.merge(tdf, on="catchment", how="left")
    predictors = list(predictorsToPretty.keys())
    predictors.remove("m")
    mask = []
    for index, row in df.iterrows():
        numNan = np.sum(row[predictors].isna())
        if numNan < 5:
            mask.append(False)
        else:
            mask.append(True)
    mask = np.array(mask)
    df = df[mask] # remove
    with open(os.path.join(logPath, "log_combiningData.txt"), "w+") as logFile:
        logFile.write("size of df prior to removal of rows with < 5 predictor features: " + str(mask.shape[0]) + "\n")
        logFile.write("size of df after removal of rows with < 5 predictor features: " + str(len(df[df.columns[0]])) + "\n")
    path = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_" + str(tag) + ".csv")
    df.to_csv(path, index=False)


def combineResults():
    # read in the metadata file(s)
    df = pd.read_csv(metadataPath)
    df = df.rename({"Catchment ID":"catchment"}, axis=1)
    df = df.drop("Unnamed: 0", axis=1)

    _combine(df, tag="raw")

    print("timeseries summaries successfully combined with metadata")


