import os
from data.metadata import *
import pandas as pd


def combineResults():
    # read in the metadata file
    df = pd.read_csv(metadataPath)

    df["catchment"] = list(range(len(df["catchment"]))) # FIXME: remove this when we have real data

    for file in os.listdir(outputFilesPath):
        if file.startswith("timeseriesSummary"):
            tdf = pd.read_csv(os.path.join(outputFilesPath, file))
            df = df.merge(tdf, on="catchment")

    path = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata.csv")
    df.to_csv(path, index=False)

    print("timeseries summaries successfully combined with metadata")


