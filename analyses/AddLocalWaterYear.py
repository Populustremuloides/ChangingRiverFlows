import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from data.metadata import *
from analyses.utilityFunctions import *


def _getMonthOfLowestFlow(flow, dates):
    '''
    helper function to determine
    which month on average has the lowest flow
    '''

    flows = np.array(flow)
    months = np.array(dates.dt.month)

    means = []
    monthsInYear = list(range(1,13))
    for month in monthsInYear:
        mask = months == month
        means.append(np.nanmean(flows[mask]))
    monthOfLowestFlow = monthsInYear[np.argmin(means)]
    return monthOfLowestFlow


def _addYearsToDf(df):
    '''
    utility function that takes the mean value for a given
    data column across all complete water years.

    Water years are calculated to start on the month that
    on average has the lowest flow, according to:

    https://doi.org/10.1029/2020WR027233

    '''

    flows = df[dischargeVar]
    dates = pd.to_datetime(df[datesVar])

    monthOfLowestFlow = _getMonthOfLowestFlow(flows, dates)
    months = np.array(dates.dt.month)
    days = np.array(dates.dt.day)

    # assign each day to a water year
    waterYear = 0
    waterYears = []
    for i in range(months.shape[0]):
        month = months[i]
        day = days[i]
        if day == 1:
            if month == monthOfLowestFlow:
                waterYear += 1
        waterYears.append(waterYear)

    df[waterYearVar] = waterYears

    # remove the incomplete water years
    waterYears, counts = np.unique(waterYears, return_counts=True)
    if counts[0] < 365:
        df = df[df[waterYearVar] != waterYears[0]]

    if counts[-1] < 365:
        df = df[df[waterYearVar] != waterYears[-1]]

    # remove the incomplete water years
    waterYears, counts = np.unique(df[waterYearVar], return_counts=True)

    return df


def addLocalWaterYear():
    '''
    Add a new column to each input flow timeseries that indicates
    the local water year, with the local water year based on the
    month of lowest flow per https://doi.org/10.1029/2020WR027233.

    New timeseries data are copied to the output directory.

    This funciton is intended to be run prior to the other
    analysis functions.
    '''


    # make the output directory if it doesn't already exist
    if not os.path.exists(augmentedTimeseriesPath):
        os.mkdir(augmentedTimeseriesPath)

    numKept = 0
    numCats = len(os.listdir(pureSeriesPath))
    loop = tqdm(total=numCats)
    for file in os.listdir(pureSeriesPath):
        dataFilePath = os.path.join(pureSeriesPath, file)
        df = pd.read_csv(dataFilePath)

        df = _addYearsToDf(df)

        if len(np.unique(df[waterYearVar])) > g_numYearsToUseForAnalysis:
            # save to new location
            df = df.drop(datesVar, axis=1)
            df.to_csv(os.path.join(augmentedTimeseriesPath, file), index=False)
            numKept += 1

        loop.set_description("Computing local water years")
        loop.update(1)

    with open(os.path.join(logPath, "log_localWaterYearLog.txt"), "w+") as logFile:
        logFile.writelines("There were a total of " + str(numCats) + " possible catchments, " + str(numKept) + " of which were kept for further analysis because they had sufficient length.")
    loop.close()
