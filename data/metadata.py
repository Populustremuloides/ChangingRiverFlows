import os
import sys

# TODO: extract just the land surface data for different catchments

# stored variable names for accessing the raw data

# global variables are labeled with g_*
g_numYearsToUseForAnalysis = 9

waterYearVar = "localWaterYear"

datesVar = "Date"

dischargeVar = "flow"
specificDischargeVar = "flow" # FIXME: this needs to be updated

tempVar = "average temperature (C)"

precipVar = "precipitation"
specificPrecipVar = "precipitation"  # FIXME: this needs to be updated

etVar = "ET [kg/m^2/8day]"
specificETVar = "ET [kg/m^2/8day]" # FIXME: this needs to be updated


petVar = "ET [kg/m^2/8day]" # FIXME: this needs to be updated
specificPETVar = "ET [kg/m^2/8day]" # FIXME: this needs to be updated
# get the current working directory
rootPath = os.getcwd()
sys.path.append(rootPath)

# get the analysis folder
analysesPath = os.path.join(rootPath, "analyses")
sys.path.append(analysesPath)

# get the data folder
dataPath = os.path.join(rootPath, "data")
#sys.path.append(dataPath)

# get the PureSeries folder
pureSeriesPath = os.path.join(dataPath, "PureSeries")
#sys.path.append(pureSeriesPath)

# get the output
outputPath = os.path.join(rootPath, "output")

# get the outputFiles
outputFilesPath = os.path.join(outputPath, "outputFiles")
augmentedTimeseriesPath = os.path.join(outputPath, "augmentedTimeseries")


print("successfully identiified directories for analysis")
