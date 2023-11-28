from data.metadata import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import string
import os

'''
The purpose of this script is to make a set of functions that
return colors used for figures in a way that is consistent
across figures.
'''



# will break if functions are called before combinedTimeseriesSummariesAndMetadata.csv is computed

loggingPath = os.path.join(logPath, "log_mappingLog.txt")
with open(loggingPath, "w+") as logFile:
    pass # reset log file


degreesC = "$^{\circ}$C"
delta = r"$\Delta$"

seismic = mpl.cm.get_cmap('seismic')
seismic_r = mpl.cm.get_cmap('seismic_r')
PiYG = mpl.cm.get_cmap('PiYG')
PiYG_r = mpl.cm.get_cmap('PiYG_r')
cool = mpl.cm.get_cmap('cool')
cool_r = mpl.cm.get_cmap('cool_r')
plasma = mpl.cm.get_cmap('plasma')
plasma_r = mpl.cm.get_cmap('plasma_r')
viridis = mpl.cm.get_cmap('viridis')
viridis_r = mpl.cm.get_cmap('viridis_r')
PuOr = mpl.cm.get_cmap("PuOr")
PuOr_r = mpl.cm.get_cmap("PuOr_r")

def getCmapFromString(cmapString):
    if cmapString == "seismic":
        cmap = mpl.cm.get_cmap('seismic')
    elif cmapString == "seismic_r":
        cmap = mpl.cm.get_cmap('seismic_r')
    elif cmapString == 'PiYG':
        cmap = mpl.cm.get_cmap('PiYG')
    elif cmapString == 'PiYG_r':
        cmap = mpl.cm.get_cmap('PiYG_r')
    elif cmapString == 'cool':
        cmap = mpl.cm.get_cmap('cool')
    elif cmapString == 'cool_r':
        cmap = mpl.cm.get_cmap('cool_r')
    elif cmapString == 'plasma':
        cmap = mpl.cm.get_cmap('plasma')
    elif cmapString == 'plasma_r':
        cmap = mpl.cm.get_cmap('plasma_r')
    elif cmapString == 'viridis':
        cmap = mpl.cm.get_cmap('viridis')
    elif cmapString == 'viridis_r':
        cmap = mpl.cm.get_cmap('viridis_r')
    elif cmapString == "PuOr":
        cmap = mpl.cm.get_cmap("PuOr")
    elif cmapString == "PuOr_r":
        cmap = mpl.cm.get_cmap("PuOr_r")
    elif cmapString == "temp":
        cmap = mpl.cm.get_cmap('seismic')
    elif cmapString == "gord":
        cmap = mpl.cm.get_cmap('PiYG')
    elif cmapString == "precip":
        cmap = sns.diverging_palette(330, 250, s=100, as_cmap=True)
    else:
        print(cmapString + " not a recognized cmap")
    return cmap


def _printTruncation(var, lowerBound, upperBound, df, transform=None):
    if transform != None:
        numTruncated = np.sum(transform(df[var]) < lowerBound) + np.sum(transform(df[var]) > upperBound)
    else:
        numTruncated = np.sum(df[var] < lowerBound) + np.sum(df[var] > upperBound)

    with open(loggingPath, "a+") as logFile:
        logFile.writelines("*******************\n")
        logFile.writelines("number truncated for " + var + " " + str(numTruncated) + "\n")
        logFile.writelines("\n")
        logFile.writelines(" = " + str(100 * (numTruncated / np.sum(~df[var].isna()))) + " % of the data\n")
        logFile.writelines("*******************\n\n")


# *********************************************************************************
# Budget Deficits
# *********************************************************************************

def getNorm_budget_deficit(df, printTruncation=True):
    var = "budget_deficit"

    #plt.hist(df[var], bins=50)
    #plt.show()
    #lowerBound = np.min(np.sort(df[var])) #FIXME: change this
    #upperBound = np.max(np.sort(df[var])) #FIXME: change this


    lowerBound = np.min(np.sort(df[var])[50:]) #FIXME: change this
    upperBound = -1 * lowerBound #np.max(np.sort(df[var])[:-10]) #FIXME: change this

    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_budget_deficit(cmap, df):
    norm = getNorm_budget_deficit(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_budget_deficit(cmap, df, save=False, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_budget_deficit(df, printTruncation=True)
    m = getM_budget_deficit(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("Budget Deficit", size=15, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_budget_deficit.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_budget_deficit.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# meanPercentDC_ModeratelyWel (mean moderately well-drained soil)
# *********************************************************************************

def getNorm_dcModeratelyWell(df, printTruncation=True):
    var = "meanPercentDC_ModeratelyWell"

    lowerBound = 0
    upperBound = 100

    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_dcModeratelyWell(cmap, df):
    norm = getNorm_dcModeratelyWell(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_dcModeratelyWell(cmap, df, save=False, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_dcModeratelyWell(df, printTruncation=True)
    m = getM_dcModeratelyWell(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("% Moderately Well-drained Soil", size=15, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_meanPercentDC_ModeratelyWell.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_meanPercentDC_ModeratelyWell.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# cls3 (proportion deciduous broadleaf trees)
# *********************************************************************************

def getNorm_cls3(df, printTruncation=True):
    var = "cls3"

    lowerBound = 0
    upperBound = 1

    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_cls3(cmap, df):
    norm = getNorm_cls3(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_M(cmap, df, save=False, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_cls3(df, printTruncation=True)
    m = getM_cls3(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("Proportion Deciduous\nBroadleaf Trees", size=15, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_cls3.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_cls3.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# cls5 (proportion shrub)
# *********************************************************************************

def getNorm_cls5(df, printTruncation=True):
    var = "cls5"

    lowerBound = 0
    upperBound = 1

    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_cls5(cmap, df):
    norm = getNorm_cls5(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_M(cmap, df, save=False, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_cls5(df, printTruncation=True)
    m = getM_cls5(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("Proportion Shrubs", size=15, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_cls5.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_cls5.png"))
    plt.clf()
    plt.close()



# *********************************************************************************
# fuh's parameter
# *********************************************************************************

def getNorm_M(df, printTruncation=True):
    var = "m"

    #plt.hist(df[var], bins=50)
    #plt.show()

    lowerBound = np.min(df[var]) #FIXME: change this
    upperBound = np.max(df[var]) #FIXME: change this

    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_M(cmap, df):
    norm = getNorm_M(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_M(cmap, df, save=False, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_M(df, printTruncation=True)
    m = getM_M(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("Fuh's Parameter", size=15, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_m.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_m.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# log mean annual precipitation
# *********************************************************************************

def transform_maspMeanLog(array):
    array = np.array(array)
    return np.log(array + 1)

def getNorm_maspMeanLog(df, printTruncation=True):
    var = "maspMean"

    lowerBound = 2 
    upperBound = np.max(transform_maspMeanLog(df[var]))

    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_maspMeanLog(cmap, df):
    norm = getNorm_maspMeanLog(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_maspMeanLog(cmap, df, save=False, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_maspMeanLog(df, printTruncation=True)
    m = getM_maspMeanLog(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("log mean annual precipitation (mm)", size=15, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_maspMeanLog.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_maspMeanLog.png"))
    plt.clf()
    plt.close()

# *********************************************************************************
# mean annual temperature
# *********************************************************************************

def getNorm_matMean(df, printTruncation=True):
    var = "matMean"
    lowerBound = np.min(df[var])
    upperBound = np.max(df[var])
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)

    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_matMean(cmap, df):
    norm = getNorm_MeanTempAnn(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_matMean(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_matMean(df)
    m = getM_matMean(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("mean annual temperature (" + degreesC + ")", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_matMean.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_matMean.png"))
    plt.clf()
    plt.close()

# *********************************************************************************
# stream order (gord)
# *********************************************************************************
def getNorm_gord(df, printTruncation=True):
    var = "gord"
    lowerBound = np.min(df[var])
    upperBound = np.max(df[var])
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)

    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_gord(cmap, df):
    norm = getNorm_gord(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_gord(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_gord(df)
    m = getM_gord(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("Strahler stream order", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_gord.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_gord.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# mean annual specific discharge - mean
# *********************************************************************************


def getNorm_masdMean(df, printTruncation=True):
    #plt.hist(df["masdMean"], bins=20)
    #plt.show()
    #quit()

    var = "masdMean"
    lowerBound = 0 
    upperBound = 5e6
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_masdMean(cmap, df):
    norm = getNorm_masdMean(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_masdMean(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_masdMean(df, printTruncation=True)
    m = getM_masdMean(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("mean annual specific discharge (L/day/km$^2$)", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_masdMean.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_masdMean.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# mean annual specific discharge - slope
# *********************************************************************************


def getNorm_masdSlope(df, printTruncation=True):
    #plt.hist(df["masdSlope"], bins=20)
    #plt.show()
    #quit()

    var = "masdSlope"
    lowerBound = -1e5 # FIXME: change
    upperBound = 1e5 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_masdSlope(cmap, df):
    norm = getNorm_masdSlope(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_masdSlope(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_masdSlope(df, printTruncation=True)
    m = getM_masdSlope(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("change in mean annual specific discharge (L/day/km$^2$ / year)", size=20, weight='bold') # FIXME: check the units
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_masdSlope.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_masdSlope.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# mean annual specific discharge - slope normalized
# *********************************************************************************


def getNorm_masdPercentChange(df, printTruncation=True):
    #plt.hist(df["masdPercentChange"], bins=20)
    #plt.show()
    #quit()

    var = "masdPercentChange"
    lowerBound = -12
    upperBound = 12
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_masdPercentChange(cmap, df):
    norm = getNorm_masdPercentChange(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_masdPercentChange(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_masdPercentChange(df, printTruncation=True)
    m = getM_masdPercentChange(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("% change in mean annual specific discharge / year", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_masdPercentChange.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_masdPercentChange.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# day of mean flow - mean
# *********************************************************************************

def getNorm_domfMean(df, printTruncation=True):
    #plt.hist(df["domfMean"], bins=20)
    #plt.show()
    #quit()

    var = "domfMean"
    lowerBound = 110#np.mean(df[var])
    upperBound = 260#np.mean(df[var])
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_domfMean(cmap, df):
    norm = getNorm_domfMean(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_domfMean(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_domfMean(df)
    m = getM_domfMean(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("day of mean flow (day in water year)", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_domfMean.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_domfMean.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# day of mean flow - slope
# *********************************************************************************


def getNorm_domfSlope(df, printTruncation=True):
    #plt.hist(df["domfSlope"], bins=20)
    #plt.show()
    #quit()    

    var = "domfSlope"
    lowerBound = -9
    upperBound = 9
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_domfSlope(cmap, df):
    norm = getNorm_domfSlope(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_domfSlope(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_domfSlope(df, printTruncation=True)
    m = getM_domfSlope(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("change in day of mean flow (days / year)", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_domfSlope.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_domfSlope.png"))
    plt.clf()
    plt.close()

# *********************************************************************************
# day of peak flow - mean
# *********************************************************************************
#plt.hist(df["dopfMean"])
#plt.clf()

def getNorm_dopfMean(df, printTruncation=True):
    #plt.hist(df["dopfMean"], bins=20)
    #plt.show()
    #quit()

    var = "dopfMean"
    lowerBound = 100#np.min(df[var])
    upperBound = 300 #np.max(df[var])
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_dopfMean(cmap, df):
    norm = getNorm_dopfMean(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_dopfMean(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_dopfMean(df)
    m = getM_dopfMean(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("day of peak flow (day in water year)", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_dopfMean.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_dopfMean.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# day of peak flow - slope
# *********************************************************************************

def getNorm_dopfSlope(df, printTruncation=True):
    #plt.hist(df["dopfSlope"])
    #plt.show()
    #quit()

    var = "dopfSlope"
    lowerBound = -20 # FIXME: change
    upperBound = 20 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_dopfSlope(cmap, df):
    norm = getNorm_dopfSlope(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_dopfSlope(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_dopfSlope(df, printTruncation=True)
    m = getM_dopfSlope(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("change in day of peak flow (days / year)", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_dopfSlope.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_dopfSlope.png"))
    plt.clf()
    plt.close()



# *********************************************************************************
# period of mean flow - mean
# *********************************************************************************
def getNorm_pommfMean(df, printTruncation=True):
    #plt.hist(df["pommfMean"], bins=20)
    #plt.show()
    #quit()

    var = "pommfMean"
    lowerBound = 20 # FIXME: change
    upperBound = 300 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_pommfMean(cmap, df):
    norm = getNorm_pommfMean(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_pommfMean(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_pommfMean(df, printTruncation=True)
    m = getM_pommfMean(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("mean period of mean flow (days)", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)
    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_pommfMean.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_pommfMean.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# period of mean flow - slope
# *********************************************************************************
def getNorm_pommfSlope(df, printTruncation=True):
    #plt.hist(df["pommfSlope"], bins=20)
    #plt.show()
    #quit()

    var = "pommfSlope"
    lowerBound = -10 # FIXME: change
    upperBound = 10 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_pommfSlope(cmap, df):
    norm = getNorm_pommfSlope(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_pommfSlope(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_pommfSlope(df, printTruncation=True)
    m = getM_pommfSlope(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("change in period of mean flow (days / year)", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_pommfSlope.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_pommfSlope.png"))
    plt.clf()
    plt.close()



# *********************************************************************************
# runoff ratio - mean
# *********************************************************************************

def getNorm_d_pMean(df, printTruncation=True):
    #plt.hist(df["d_pMean"], bins=10)
    #plt.show()
    #quit()

    var = "d_pMean"
    lowerBound = 0 # FIXME: change
    upperBound = 1.5 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_d_pMean(cmap, df):
    norm = getNorm_d_pMean(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_d_pMean(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_d_pMean(df, printTruncation=True)
    m = getM_d_pMean(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("runoff ratio", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_d_pMean.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_d_pMean.png"))
    plt.clf()
    plt.close()


# *********************************************************************************
# runoff ratio - slope
# *********************************************************************************

def getNorm_d_pSlope(df, printTruncation=True):
    #plt.hist(df["d_pSlope"], bins=20)
    #plt.show()
    #quit()

    var = "d_pSlope"
    lowerBound = -0.03 # FIXME: change
    upperBound = 0.03 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_d_pSlope(cmap, df):
    norm = getNorm_d_pSlope(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_d_pSlope(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_d_pSlope(df, printTruncation=True)
    m = getM_d_pSlope(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("change in runoff ratio / year", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_d_pSlope.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_d_pSlope.png"))
    plt.clf()
    plt.close()

# *********************************************************************************
# runoff ratio - percent change
# *********************************************************************************

def getNorm_d_pPercentChange(df, printTruncation=True):
    #plt.hist(df["d_pPercentChange"], bins=20)
    #plt.show()
    #quit()

    var = "d_pPercentChange"
    lowerBound = -10 # FIXME: change
    upperBound = 10 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, df)

    return norm

def getM_d_pPercentChange(cmap, df):
    norm = getNorm_d_pPercentChange(df)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_d_pPercentChange(cmap, df, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_d_pPercentChange(df, printTruncation=True)
    m = getM_d_pPercentChange(cmap, df)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("% change in runoff ratio / year", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if pLeft:
        plt.savefig(os.path.join(figurePath,"colorbarL_d_pPercentChange.png"))
    else:
        plt.savefig(os.path.join(figurePath,"colorbar_d_pPercentChange.png"))
    plt.clf()
    plt.close()


# ***********************************************************************************


def getColors(var, m, df, transform=None):
    vals = df[var]
    lower, upper = m.get_clim()

    if transform != None:
        vals = transform(vals)

    colors = []
    for val in vals:
        c = m.to_rgba(val)
        colors.append(c)
    return colors

def getM(variable, cmap, df):
    if variable == "maspMeanLog":
        function = getM_MeanPrecAnnLog
    elif variable == "matMean":
        function = getM_MeanTempAnn
    elif variable == "gord":
        function = getM_gord
    elif variable == "masdMean":
        function = getM_masdMean
    elif variable == "masdSlope":
        function = getM_masdSlope
    elif variable == "masdPercentChange":
        function = getM_masdPercentChange
    elif variable == "domfMean":
        function = getM_domfMean
    elif variable == "domfSlope":
        function = getM_domfSlope
    elif variable == "dopfMean":
        function = getM_domfMean
    elif variable == "dopfSlope":
        function = getM_domfSlope
    elif variable == "pommfMean":
        function = getM_pommfMean
    elif variable == "pommfSlope":
        function = getM_pommfSlope
    elif variable == "d_pMean":
        function = getM_d_pMean
    elif variable == "d_pSlope":
        function = getM_d_pSlope
    elif variable == "d_pPercentChange":
        function = getM_d_pPercentChange
    elif variable == "m":
        function = getM_M
    elif variable == "budget_deficit":
        function = getM_budget_deficit
    elif variable == "cls3":
        function = getM_cls3
    elif variable == "cls5":
        function = getM_cls5
    elif variable == "meanPercentDC_ModeratelyWell":
        function = getM_dcModeratelyWell
    else:
        print(variable, " not recognized as a variable that can be used to color catchments")

    if type(cmap) == type("string"):
        cmap = getCmapFromString(cmap)

    m = function(cmap, df)
    return m

def plotColorbar(variable, cmap, df, pLeft=False):
    if variable == "maspMeanLog":
        colorbar_MeanPrecAnnLog(cmap, df, pLeft=pLeft)
    elif variable == "matMean":
        colorbar_MeanTempAnn(cmap, df, pLeft=pLeft)
    elif variable == "gord":
        colorbar_gord(cmap, df, pLeft=pLeft)
    elif variable == "masdMean":
        colorbar_masdMean(cmap, df, pLeft=pLeft)
    elif variable == "masdSlope":
        colorbar_masdSlope(cmap, df, pLeft=pLeft)
    elif variable == "masdPercentChange":
        colorbar_masdSlopeNormalized(cmap, df, pLeft=pLeft)
    elif variable == "domfMean":
        colorbar_domfMean(cmap, df, pLeft=pLeft)
    elif variable == "domfSlope":
        colorbar_dopfSlope(cmap, df, pLeft=pLeft)
    elif variable == "dopfMean":
        colorbar_dopfMean(cmap, df, pLeft=pLeft)
    elif variable == "dopfSlope":
        colorbar_domfSlope(cmap, df, pLeft=pLeft)
    elif variable == "pommfMean":
        colorbar_pommfMean(cmap, df, pLeft=pLeft)
    elif variable == "pommfSlope":
        colorbar_pommfSlope(cmap, df, pLeft=pLeft)
    elif variable == "d_pMean":
        colorbar_d_pMean(cmap, df, pLeft=pLeft)
    elif variable == "d_pSlope":
        colorbar_d_pSlope(cmap, df, pLeft=pLeft)
    elif variable == "d_pPercentChange":
        colorbar_d_pPercentChange(cmap, df, pLeft=pLeft)
    elif variable == "m":
        colorbar_M(cmap, df, pLeft=pLeft)
    elif variable == "budget_deficit":
        colorbar_budget_deficit(cmap, df, pLeft=pLeft)
    elif variable == "cls3":
        colorbar_cls3(cmap, df, pLeft=pLeft)
    elif variable == "cls5":
        colorbar_cls5(cmap, df, pLeft=pLeft)
    elif variable == "meanPercentDC_ModeratelyWell":
        colorbar_dcModeratelyWell(cmap, df, pLeft=pLeft)

    else:
        print(variable, " not recognized as a variable that can be used to color catchments")

def getTransform(variable):
    if variable == "maspMeanLog":
        transform = transform_maspMeanLog
    else:
        transform = None
    return transform


