import pandas as pd
import os
from data.metadata import *


def combineTimeseries():
    '''
    takes the data from the normalized and non-normalized
    timeseries joints them together
    '''


    # grab the un-normalized file names
    litersPerDayFiles = []
    for file in os.listdir(litersPerDayPath):
        if file.endswith("csv"):
            litersPerDayFiles.append(file)

    # grab the normalized file names
    litersPerDayPerSqKmFiles = []
    for file in os.listdir(litersPerDayPerSqKmPath):
        if file.endswith("csv"):
            litersPerDayPerSqKmFiles.append(file)

    # keep only the intersection of them
    litersPerDayPerSqKmFiles = set(litersPerDayPerSqKmFiles)
    litersPerDayFiles = set(litersPerDayFiles)
    files = list(litersPerDayFiles.intersection(litersPerDayPerSqKmFiles))
    files.sort()


    # prepare the output folder
    if not os.path.exists(pureSeriesPath):
        os.mkdir(pureSeriesPath)

    for file in files:
        # read in both the unormalized and normalized data
        lpdPath = os.path.join(litersPerDayPath, file)
        lpdpsqPath = os.path.join(litersPerDayPerSqKmPath, file)

        lqdDf = pd.read_csv(lpdPath)
        lqdpsqDf = pd.read_csv(lpdpsqPath)

        # label the columns appropriately
        lqdDf = lqdDf.rename({"flow":dischargeVar,"average temperature (C)":tempVar,"precipitation":precipVar,"ET":etVar,"PET":petVar}, axis=1)
        lqdDf = lqdDf.drop(tempVar, axis=1)
        lqdpsqDf = lqdpsqDf.rename({"flow":specificDischargeVar,"average temperature (C)":tempVar,"precipitation":specificPrecipVar,"ET":specificETVar,"PET":specificPETVar}, axis=1)

        # combine the data
        tDf = lqdDf.merge(lqdpsqDf, on=datesVar)

        # save the data
        tDf.to_csv(os.path.join(pureSeriesPath, file), index=False)


