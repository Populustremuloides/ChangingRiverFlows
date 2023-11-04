import os
from data.metadata import *
import pandas as pd

def _combine(df, tag):

    for file in os.listdir(outputFilesPath):
        if file.startswith("timeseriesSummary"):

            tdf = pd.read_csv(os.path.join(outputFilesPath, file))
            df = df.merge(tdf, on="catchment")

    path = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_" + str(tag) + ".csv")
    df.to_csv(path, index=False)


def combineResults():
    # read in the metadata file(s)
    df = pd.read_csv(metadataPath)
    df = df.rename({"Catchment ID":"catchment"}, axis=1)
    df = df.drop("Unnamed: 0", axis=1)

    imputedDataPath = os.path.join(outputFilesPath, "imputedMetadata.csv")
    imputedDf = pd.read_csv(imputedDataPath)
    imputedDf = imputedDf.rename({"Catchment ID":"catchment"}, axis=1)

    imputedPCADataPath = os.path.join(outputFilesPath, "imputedMetadataPCA.csv")
    imputedPCADf = pd.read_csv(imputedPCADataPath)
    imputedPCADf = imputedPCADf.rename({"Catchment ID":"catchment"}, axis=1)

    _combine(df, tag="raw")
    _combine(imputedDf, tag="imputed")
    _combine(imputedPCADf, tag="imputedPCA")

    print("timeseries summaries successfully combined with metadata")


