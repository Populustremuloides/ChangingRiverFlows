from scipy.stats import theilslopes
import numpy as np
from data.metadata import *

from scipy.signal import blackman
# utility functions start with u_*

def u_regressionFunction(x, y):
    res = theilslopes(y, x=x)
    return res[0]

#u_regressionFunction = theilslopes

def u_getCatchmentName(fileName):
    cat = fileName.split(".")[0]
    cat = cat.split("_")[-1]
    return cat

def u_getDayOfMeanMagnitude(timeseries):
    dayInWaterYear = np.arange(timeseries.shape[0])
    dayOfMeanMagnitude = np.sum(dayInWaterYear * timeseries) / np.sum(timeseries) #timeseries.shape[0]
    return dayOfMeanMagnitude

def u_getPeriodOfMeanMagnitude(timeseries):
    timeseries = timeseries / np.max(timeseries) # normalize
    timeseries = timeseries - np.mean(timeseries) # subtract the mean

    numDays = timeseries.shape[0]
    dt = 1. # one day per measurement
    periods = 1. / np.fft.rfftfreq(numDays, d=dt)[1:] # ignore the infinite time-horizon element

    w = blackman(timeseries.shape[0]) # use a blackman filter to reduce spectral leakage
    result = np.fft.rfft(timeseries * w)
    magnitudes = np.abs(result) ** 2

    magnitudes = magnitudes[1:] # ignore the infinite time horizon element

    periodOfMeanMagnitude = np.sum(periods * magnitudes) / np.sum(magnitudes) #periods.shape[0]

    return periodOfMeanMagnitude
