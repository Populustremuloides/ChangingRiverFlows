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

dfPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata.csv")
df = pd.read_csv(dfPath)

logPath = os.path.join(outputFilesPath, "mappingLog.txt")
with open(logPath, "w+") as logFile:
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


def _printTruncation(var, lowerBound, upperBound, transform=None):
    if transform != None:
        numTruncated = np.sum(transform(df[var]) < lowerBound) + np.sum(transform(df[var]) > upperBound)
    else:
        numTruncated = np.sum(df[var] < lowerBound) + np.sum(df[var] > upperBound)

    with open(logPath, "a+") as logFile:
        logFile.writelines("*******************")
        logFile.writelines("number truncated for " + var + " " + str(numTruncated))
        logFile.writelines("")
        logFile.writelines(" = " + str(100 * (numTruncated / np.sum(~df[var].isna()))) + " % of the data")
        logFile.writelines("*******************")

# *********************************************************************************
# log mean annual precipitation
# *********************************************************************************

def transform_maspMeanLog(array):
    array = np.array(array)
    return np.log(array + 1)

def getNorm_maspMeanLog(printTruncation=False):
    var = "maspMean"
    norm = mpl.colors.Normalize(vmin=2, vmax=np.max(transform_maspMeanLog(df[var])))
    lowerBound = 2 #FIXME: change this
    upperBound = np.inf #FIXME: change this
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)

    return norm

def getM_maspMeanLog(cmap):
    norm = getNorm_maspMeanLog()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_maspMeanLog(cmap, save=False, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_maspMeanLog(printTruncation=True)
    m = getM_maspMeanLog(cmap)
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

def getNorm_matMean(printTruncation=False):
    norm = mpl.colors.Normalize(vmin=np.min(df["matMean"]), vmax=np.max(df["matMean"]))
    return norm

def getM_matMean(cmap):
    norm = getNorm_MeanTempAnn()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_matMean(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_matMean()
    m = getM_matMean(cmap)
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
def getNorm_gord(printTruncation=False):
    norm = mpl.colors.Normalize(vmin=np.min(df["gord"]), vmax=np.max(df["gord"]))
    return norm

def getM_gord(cmap):
    norm = getNorm_gord()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_gord(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_gord()
    m = getM_gord(cmap)
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
#plt.hist(df["masdMean"])
#plt.clf()

def getNorm_masdMean(printTruncation=False):
    var = "masdMean"
    lowerBound = -1 # FIXME: change
    upperBound = 1 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)
    return norm

def getM_masdMean(cmap):
    norm = getNorm_masdMean()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_masdMean(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_masdMean(printTruncation=True)
    m = getM_masdMean(cmap)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("mean annual specific discharge (L/s/km$^2$)", size=20, weight='bold')
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
#plt.hist(df["masdSlope"])
#plt.clf()

def getNorm_masdSlope(printTruncation=False):
    var = "masdSlope"
    lowerBound = -1 # FIXME: change
    upperBound = 1 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)

    return norm

def getM_masdSlope(cmap):
    norm = getNorm_masdSlope()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_masdSlope(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_masdSlope(printTruncation=True)
    m = getM_masdSlope(cmap)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("change in mean annual specific discharge (L/s/km$^2$ / year)", size=20, weight='bold') # FIXME: check the units
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
#plt.hist(df["masdPercentChange"])
#plt.clf()

def getNorm_masdPercentChange(printTruncation=False):
    var = "masdPercentChange"
    lowerBound = -1 # FIXME: change
    upperBound = 1 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)

    return norm

def getM_masdPercentChange(cmap):
    norm = getNorm_masdPercentChange()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_masdPercentChange(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_masdPercentChange(printTruncation=True)
    m = getM_masdPercentChange(cmap)
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
#plt.hist(df["domfMean"])
#plt.clf()

def getNorm_domfMean():
    domf_means = df["domfMean"]
    norm = mpl.colors.Normalize(vmin=np.min(domf_means), vmax=np.max(domf_means))
    return norm

def getM_domfMean(cmap):
    norm = getNorm_domfMean()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_domfMean(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_domfMean()
    m = getM_domfMean(cmap)
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
#plt.hist(df["domfSlope"])
#plt.clf()

def getNorm_domfSlope(printTruncation=False):
    var = "domfSlope"
    lowerBound = -1 # FIXME: change
    upperBound = 1 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
       _printTruncation(var, lowerBound, upperBound)

    return norm

def getM_domfSlope(cmap):
    norm = getNorm_domfSlope()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_domfSlope(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_domfSlope(printTruncation=True)
    m = getM_domfSlope(cmap)
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

def getNorm_dopfMean():
    domf_means = df["dopfMean"]
    norm = mpl.colors.Normalize(vmin=np.min(domf_means), vmax=np.max(domf_means))
    return norm

def getM_dopfMean(cmap):
    norm = getNorm_dopfMean()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_dopfMean(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_dopfMean()
    m = getM_dopfMean(cmap)
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
#plt.hist(df["dopfSlope"])
#plt.clf()

def getNorm_dopfSlope(printTruncation=False):
    var = "dopfSlope"
    lowerBound = -1 # FIXME: change
    upperBound = 1 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
       _printTruncation(var, lowerBound, upperBound)

    return norm

def getM_dopfSlope(cmap):
    norm = getNorm_dopfSlope()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_dopfSlope(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_dopfSlope(printTruncation=True)
    m = getM_dopfSlope(cmap)
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
#plt.hist(df["pommfMean"])
#plt.clf()

def getNorm_pommfMean(printTruncation=False):
    var = "pommfMean"
    lowerBound = -1 # FIXME: change
    upperBound = 1 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)

    return norm

def getM_pommfMean(cmap):
    norm = getNorm_pommfMean()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_pommfMean(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_pommfMean(printTruncation=True)
    m = getM_pommfMean(cmap)
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
#plt.hist(df["pommfSlope"])
#plt.clf()

def getNorm_pommfSlope(printTruncation=False):
    var = "pommfSlope"
    lowerBound = -1 # FIXME: change
    upperBound = 1 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
       _printTruncation(var, lowerBound, upperBound)

    return norm

def getM_pommfSlope(cmap):
    norm = getNorm_pommfSlope()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_pommfSlope(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_pommfSlope(printTruncation=True)
    m = getM_pommfSlope(cmap)
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
#plt.hist(df["d_pMean"])
#plt.clf()

def getNorm_d_pMean(printTruncation=False):
    var = "d_pMean"
    lowerBound = -1 # FIXME: change
    upperBound = 1 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
       _printTruncation(var, lowerBound, upperBound)

    return norm

def getM_d_pMean(cmap):
    norm = getNorm_d_pMean()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_d_pMean(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_d_pMean(printTruncation=True)
    m = getM_d_pMean(cmap)
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
#plt.hist(df["d_Slope"])
#plt.clf()

def getNorm_d_pSlope(printTruncation=False):
    var = "d_pSlope"
    lowerBound = -1 # FIXME: change
    upperBound = 1 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
       _printTruncation(var, lowerBound, upperBound)

    return norm

def getM_d_pSlope(cmap):
    norm = getNorm_d_pSlope()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_d_pSlope(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_d_pSlope(printTruncation=True)
    m = getM_d_pSlope(cmap)
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
#plt.hist(df["d_pPercentChange"])
#plt.clf()

def getNorm_d_pPercentChange(printTruncation=False):
    var = "d_pPercentChange"
    lowerBound = -1 # FIXME: change
    upperBound = 1 # FIXME: change
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
       _printTruncation(var, lowerBound, upperBound)

    return norm

def getM_d_pPercentChange(cmap):
    norm = getNorm_d_pPercentChange()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_d_pPercentChange(cmap, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_d_pPercentChange(printTruncation=True)
    m = getM_d_pPercentChange(cmap)
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

def getM(variable, cmap):
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
    else:
        print(variable, " not recognized as a variable that can be used to color catchments")

    if type(cmap) == type("string"):
        cmap = getCmapFromString(cmap)

    m = function(cmap)
    return m

def plotColorbar(variable, cmap, pLeft=False):
    if variable == "maspMeanLog":
        colorbar_MeanPrecAnnLog(cmap, pLeft=pLeft)
    elif variable == "matMean":
        colorbar_MeanTempAnn(cmap, pLeft=pLeft)
    elif variable == "gord":
        colorbar_gord(cmap, pLeft=pLeft)
    elif variable == "masdMean":
        colorbar_masdMean(cmap, pLeft=pLeft)
    elif variable == "masdSlope":
        colorbar_masdSlope(cmap, pLeft=pLeft)
    elif variable == "masdPercentChange":
        colorbar_masdSlopeNormalized(cmap, pLeft=pLeft)
    elif variable == "domfMean":
        colorbar_domfMean(cmap, pLeft=pLeft)
    elif variable == "domfSlope":
        colorbar_dopfSlope(cmap, pLeft=pLeft)
    elif variable == "dopfMean":
        colorbar_dopfMean(cmap, pLeft=pLeft)
    elif variable == "dopfSlope":
        colorbar_domfSlope(cmap, pLeft=pLeft)
    elif variable == "pommfMean":
        colorbar_pommfMean(cmap, pLeft=pLeft)
    elif variable == "pommfSlope":
        colorbar_pommfSlope(cmap, pLeft=pLeft)
    elif variable == "d_pMean":
        colorbar_d_pMean(cmap, pLeft=pLeft)
    elif variable == "d_pSlope":
        colorbar_d_pSlope(cmap, pLeft=pLeft)
    elif variable == "d_pPercentChange":
        colorbar_d_pPercentChange(cmap, pLeft=pLeft)

    else:
        print(variable, " not recognized as a variable that can be used to color catchments")

def getTransform(variable):
    if variable == "maspMeanLog":
        transform = transform_maspMeanLog
    else:
        transform = None
    return transform


