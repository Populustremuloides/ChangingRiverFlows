import os
import sys
import numpy as np
np.random.seed(0)

# stored variable names for accessing the raw data

# global variables are labeled with g_*
g_numYearsToUseForAnalysis = 8
g_numPCAVarsToKeep = 14 # number of principal components to keep, an admittedly arbitrary number

waterYearVar = "localWaterYear"

datesVar = "Date"

dischargeVar = "discharge"
specificDischargeVar = "specificDischarge"

tempVar = "meanTemp"

precipVar = "precip"
specificPrecipVar = "specificPrecip"

etVar = "et"
specificETVar = "specificET"
petVar = "pet"
specificPETVar = "specificPET"

# get the current working directory
rootPath = os.getcwd()
sys.path.append(rootPath)

# get the analysis folder
analysesPath = os.path.join(rootPath, "analyses")
sys.path.append(analysesPath)

# get the data folder
dataPath = os.path.join(rootPath, "data")
metadataPath = os.path.join(dataPath, "metadata.csv")


# get the PureSeries folder
pureSeriesPath = os.path.join(dataPath, "PureSeries")
litersPerDayPath = os.path.join(dataPath, "FilledFinalSeries_LitersPerDay")
litersPerDayPerSqKmPath = os.path.join(dataPath, "FilledFinalSeries_LitersPerDayPerSqKm")

# get the output
outputPath = os.path.join(rootPath, "output")
if not os.path.exists(outputPath):
    os.mkdir(outputPath)



# get the outputFiles
outputFilesPath = os.path.join(outputPath, "outputFiles")
if not os.path.exists(outputFilesPath):
    os.mkdir(outputFilesPath)

augmentedTimeseriesPath = os.path.join(outputPath, "augmentedTimeseries")
if not os.path.exists(augmentedTimeseriesPath):
    os.mkdir(augmentedTimeseriesPath)

figurePath = os.path.join(outputPath, "figures")
if not os.path.exists(figurePath):
    os.mkdir(figurePath)

individualVarsPath = os.path.join(figurePath, "individualVars")
if not os.path.exists(individualVarsPath):
    os.mkdir(individualVarsPath)

# make a log directory
logPath = os.path.join(rootPath, "logs")
if not os.path.exists(logPath):
    os.mkdir(logPath)


print("successfully identiified directories for analysis")


predictablesToPretty = {
    "domfSlope":"yearly change\nin day of\nmean flow",
    "dopfSlope":"yearly change\nin day of\npeak flow",
    "masdSlope":"yearly change\nin mean annual\nspecific discharge",
    "masdPercentChange":"yearly % change\nin mean annual\nspecific discharge",
    "pommfSlope":"yearly change\nin period of\nmean flow",
    "d_pSlope":"yearly change\nin runoff ratio",
    "d_pPercentChange":"yearly % change\nin runoff ratio",
}

# ways of getting at human extractive water use
# changes in flow vs changes in runoff ratio vs changes in pet / et ratios

predictorsToPretty = {
    "p_petSlope":"yearly change in precipitation / potential evapotranspiration",
    "p_petPercentChange":"yearly change in precipitation / potential evapotranspiration",
    "dompSlope":"yearly change in day of mean precipitation",
    "maspSlope":"mean annual specific precipitation",
    "maspPercentChange":"yearly % change in mean annual specific precipitation",
    "p_petMean":"mean value of precip / potential evapotranspiration",
    "maspMean":"mean value of mean annual specific precipitation",
    "dompMean":"mean value of day of mean precipitation",
    "pet_pSlope":"yearly change in potential evapotranspiration / precipitation",
    "pet_pPercentChange":"yearly % change in potential evapotranspiration / precipitation",
    "doppetMean":"mean value of day of peak potential evapotranspiration",
    "pet_etSlope":"yearly change in potential evapotranspiration / evapotranspiration",
    "pet_etPercentChange":"yearly % change in potential evapotranspiration / evapotranspiration",
    "dompetSlope":"yearly change in day of mean potential evapotranspiration",
    "maspetSlope":"yearly change in mean annual specific evapotranspiration",
    "maspetPercentChange":"yearly % change in mean annual specific evapotranspiration",
    "et_pSlope":"yearly change in evapotranspiration / precipitation",
    "et_pPercentChange":"yearly % change in evapotranspiration / precipitation",
    "dometSlope":"yearly change in day of mean evapotranspiration",
    "masetSlope":"yearly change in mean annual specific evapotranspiration",
    "masetPercentChange":"yearly % change in mean annual specific evapotranspiration",
    "pet_pMean":"mean value of potential evapotranspiration / precipitation",
    "pet_etMean":"mean value of potential evapotranspiration / evapotranspiration",
    "dompetMean":"mean value of day of mean potential evapotranspiration",
    "maspetMean":"mean value of mean annual specific potential evapotranspiration",
    "d_pMean":"mean value of runoff ratio",
    "masdMean":"mean vaue of mean annual speicifc discharge",
    "domfMean":"mean value of day of mean flow",
    "et_pMean":"mean value of evepotranspiration / precipitation",
    "dometMean":"mean value of day of mean evapotranspiration",
    "masetMean":"mean value of mean annual specific evapotranspiration",
    "dopfMean":"mean value of day of peak flow",
    "doppMean":"mean value of day of peak precipitation",
    "doptMean":"mean value of day of peak temperature",
    "pommfMean":"mean value of period of mean flow",
    "pommpMean":"mean value of period of mean precipitation",
    "pommtSlope":"yearly change in period of mean temperature",
    "doppetSlope":"yearly change in day of peak potential pevapotranspiration",
    "matSlope":"yearly change in mean annual temperature",
    "domtSlope":"yearly change in day of mean temperature",
    "doppSlope":"yearly change in day of peak precipitation",
    "doptSlope":"yearly change in day of peak temperature",
    "pommpetSlope":"yearly change in period of mean potential evapotranspiration",
    "pommpSlope":"yearly change in day of mean precipitation",
    "pommpetMean":"mean value of day of mean potential evapotranspiration",
    "dopetSlope":"day of peak evapotranspiration",
    "matMean":"mean value of mean annual temperature",
    "domtMean":"mean value of day of mean temperature",
    "pommetMean":"mean value of period of mean evapotranspiration",
    "dopetMean":"mean value of day of peak evapotranspiration",
    "pommtMean":"mean value of period of mean temperature",
    "pommetSlope":"yearly change in period of mean evapotranspiration",
    "m":"fuh's parameter",
    "cls1":"proportion evergreen deciduous trees",
    "cls2":"proportion braodleaf evergree trees",
    "cls3":"proportion deciduous broadleaf trees",
    "cls4":"proportion mixed other trees",
    "cls5":"proportion shrubs",
    "cls6":"proportion herbacious vegetation",
    "cls7":"proportion cultivated and managed vegetation",
    "cls8":"proportion regularly flooded vegetation",
    "cls9":"proportion urbat",
    "cls10":"proportion snow/ice",
    "cls11":"proportion barren",
    "cls12":"proportion open water",

    "Dam_SurfaceArea":"total surface area of dams",
    "Dam_Count":"total number of dams",
    "MeanPopden_2000":"mean population density in year 2000",
    "MeanPopden_2005":"mean population density in year 2005",
    "MeanPopden_2010":"mean population density in year 2010",
    "MeanPopden_2015":"mean population density in year 2015",
    "MeanHumanFootprint":"mean human footprint",

    "gord":"strahler stream order",
    "PathLength":"length of longest path in watershed",
    "TotalLength":"total length of all paths in watershed",
    "drain_den":"drainage density",

    "meanPercentDC_Well":"percent well drained",
    "meanPercentDC_VeryPoor":"percent very poorly drained",
    "meanPercentDC_SomewhatExcessive":"percent somewhat excessively drained",
    "meanPercentDC_Poor":"percent poorly drained",
    "meanPercentDC_ModeratelyWell":"percent moderately well drained",
    "meanPercentDC_Imperfectly":"percent imperfectly drained"
}


predictorsToPrettyPCA = {
    "p_petSlope":"yearly change in precipitation / potential evapotranspiration",
    "p_petPercentChange":"yearly change in precipitation / potential evapotranspiration",
    "dompSlope":"yearly change in day of mean precipitation",
    "maspSlope":"mean annual specific precipitation",
    "maspPercentChange":"yearly % change in mean annual specific precipitation",
    "p_petMean":"mean value of precip / potential evapotranspiration",
    "maspMean":"mean value of mean annual specific precipitation",
    "dompMean":"mean value of day of mean precipitation",
    "pet_pSlope":"yearly change in potential evapotranspiration / precipitation",
    "pet_pPercentChange":"yearly % change in potential evapotranspiration / precipitation",
    "doppetMean":"mean value of day of peak potential evapotranspiration",
    "pet_etSlope":"yearly change in potential evapotranspiration / evapotranspiration",
    "pet_etPercentChange":"yearly % change in potential evapotranspiration / evapotranspiration",
    "dompetSlope":"yearly change in day of mean potential evapotranspiration",
    "maspetSlope":"yearly change in mean annual specific evapotranspiration",
    "maspetPercentChange":"yearly % change in mean annual specific evapotranspiration",
    "et_pSlope":"yearly change in evapotranspiration / precipitation",
    "et_pPercentChange":"yearly % change in evapotranspiration / precipitation",
    "dometSlope":"yearly change in day of mean evapotranspiration",
    "masetSlope":"yearly change in mean annual specific evapotranspiration",
    "masetPercentChange":"yearly % change in mean annual specific evapotranspiration",
    "pet_pMean":"mean value of potential evapotranspiration / precipitation",
    "pet_etMean":"mean value of potential evapotranspiration / evapotranspiration",
    "dompetMean":"mean value of day of mean potential evapotranspiration",
    "maspetMean":"mean value of mean annual specific potential evapotranspiration",
    "d_pMean":"mean value of runoff ratio",
    "masdMean":"mean vaue of mean annual speicifc discharge",
    "domfMean":"mean value of day of mean flow",
    "et_pMean":"mean value of evepotranspiration / precipitation",
    "dometMean":"mean value of day of mean evapotranspiration",
    "masetMean":"mean value of mean annual specific evapotranspiration",
    "dopfMean":"mean value of day of peak flow",
    "doppMean":"mean value of day of peak precipitation",
    "doptMean":"mean value of day of peak temperature",
    "pommfMean":"mean value of period of mean flow",
    "pommpMean":"mean value of period of mean precipitation",
    "pommtSlope":"yearly change in period of mean temperature",
    "doppetSlope":"yearly change in day of peak potential pevapotranspiration",
    "matSlope":"yearly change in mean annual temperature",
    "domtSlope":"yearly change in day of mean temperature",
    "doppSlope":"yearly change in day of peak precipitation",
    "doptSlope":"yearly change in day of peak temperature",
    "pommpetSlope":"yearly change in period of mean potential evapotranspiration",
    "pommpSlope":"yearly change in day of mean precipitation",
    "pommpetMean":"mean value of day of mean potential evapotranspiration",
    "dopetSlope":"day of peak evapotranspiration",
    "matMean":"mean value of mean annual temperature",
    "domtMean":"mean value of day of mean temperature",
    "pommetMean":"mean value of period of mean evapotranspiration",
    "dopetMean":"mean value of day of peak evapotranspiration",
    "pommtMean":"mean value of period of mean temperature",
    "pommetSlope":"yearly change in period of mean evapotranspiration",
    "m":"fuh's parameter",
    "1":"Low Winter Temperature",
    "2":"Large Catchment Area and Dam Count",
    "3":"High Precipitation",
    "4":"Low Summer Temperature, Low Human Presence, Poor Soil Drainage",
    "5":"High Agricultural Density, High Population Density, Low Forest Cover",
    "6":"High Population Density, Low Agricutlural Density, High Forest Cover",
    "7":"Wet Winters and Evergreen Needle Plants",
    "8":"High Precipitation Seasonality",
    "9":"Small, Cultivated or Inhabited Catchments with Wet Winters",
    "10":"Wet Catchments with Few but Potential Large Dams or Lakes",
    "11":"Small Catchments with High Lake and Dam Area",
    "12":"Barren Shrubland or Snow, Higher Altitude, Steeper Catchments with Poor Draining Soil",
    "13":"Barren, Snow, or Deciduous Cover with Well-draining Soil and Larger Catchment Area",
    "14":"Small, Regularly Flooded Catchments with Few Dams, Evergreen Deciduous Trees, Poorly Draining Soil, and High Density"
}



predictorsToCategory = {
    "p_petSlope":"climate fluctuation",
    "p_petPercentChange":"climate fluctuation",
    "dompSlope":"climate fluctuation",
    "maspSlope":"climate fluctuation",
    "maspPercentChange":"climate fluctuation",
    "p_petMean":"climate mean",
    "maspMean":"climate mean",
    "dompMean":"climate mean",
    "pet_pSlope":"climate fluctuation",
    "pet_pPercentChange":"climate fluctuation",
    "doppetMean":"climate mean",
    "pet_etSlope":"climate fluctuation",
    "pet_etPercentChange":"climate fluctuation",
    "dompetSlope":"climate fluctuation",
    "maspetSlope":"climate fluctuation",
    "maspetPercentChange":"climate fluctuation",

    "et_pSlope":"climate mean",
    "et_pPercentChange":"climate fluctuation",
    "dometSlope":"climate fluctuation",
    "masetSlope":"climate fluctuation",
    "masetPercentChange":"climate fluctuation",
    "pet_pMean":"climate mean",
    "pet_etMean":"climate mean",
    "dompetMean":"climate mean",
    "maspetMean":"climate mean",
    "d_pMean":"mean flow properties",
    "masdMean":"mean flow properties",
    "domfMean":"mean flow properties",
    "et_pMean":"climate mean",
    "dometMean":"climate mean",
    "masetMean":"climate mean",
    "dopfMean":"mean flow properties",
    "doppMean":"climate mean",
    "doptMean":"climate mean",
    "pommfMean":"mean flow properties",
    "pommpMean":"climate mean",
    "pommtSlope":"climate fluctuation",
    "doppetSlope":"climate fluctuation",
    "matSlope":"climate fluctuation",
    "domtSlope":"climate fluctuation",
    "doppSlope":"climate fluctuation",
    "doptSlope":"climate fluctuation",
    "pommpetSlope":"climate fluctuation",
    "pommpSlope":"climate fluctuation",
    "pommpetMean":"climate mean",
    "dopetSlope":"climate fluctuation",
    "matMean":"climate mean",
    "domtMean":"climate mean",
    "pommetMean":"climate mean",
    "dopetMean":"climate mean",
    "pommtMean":"climate mean",
    "pommetSlope":"climate fluctuation",
    "m":"fuh's parameter",

    "cls1":"land cover",
    "cls2":"land cover",
    "cls3":"land cover",
    "cls4":"land cover",
    "cls5":"land cover",
    "cls6":"land cover",
    "cls7":"agriculture",
    "cls8":"land cover",
    "cls9":"population density",
    "cls10":"land cover",
    "cls11":"land cover",
    "cls12":"land cover",

    "Dam_SurfaceArea":"dams",
    "Dam_Count":"dams",
    "MeanPopden_2000":"population density",
    "MeanPopden_2005":"population density",
    "MeanPopden_2010":"population density",
    "MeanPopden_2015":"population density",
    "MeanHumanFootprint":"population density",

    "gord":"size",
    "PathLength":"size",
    "TotalLength":"size",
    "gelev_m":"elevation",
    "drain_den":"drainage properties",

    "meanPercentDC_Well":"drainage properties",
    "meanPercentDC_VeryPoor":"drainage properties",
    "meanPercentDC_SomewhatExcessive":"drainage properties",
    "meanPercentDC_Poor":"drainage properties",
    "meanPercentDC_ModeratelyWell":"drainage properties",
    "meanPercentDC_Imperfectly":"drainage properties"
}

predictorsToCategoryPCA = {
    "1":"High winter temperature, high winter precipitation",
    "2":"Large catchment area, dam area, and dam count. Later seasonal precip. and (P)ET",
    "3":"High precipitation, high P/PET ratio, large area",
    "4":"Low summer remperature, low human presence, high altitude",
    "5":"Large increases in P/(P)ET ratio, lower summer temperature",
    "6":"Low forest cover, high human population, and high agricultural land cover",
    "7":"increase in ET/PET ratio, high population density",
    "8":"decrease in ET/PET ratio, high population density",
    "9":"Little open water, little seasonal pattern to Precip. or (P)ET, late arrival of peak flows/precip.",
    "10":"High PET/ET ratio, lots of open water, poorly draining soil, many large dams",
    "11":"High human agriculture levels, seasonal precip., few dams and little forest cover",
    "12":"Changes to later peak in precip. and ET levels, multiannual climate cycles, broadleaf evergreen trees",
    "13":"Low potential evapotranspiration, deciduous broadleaf trees, poorly draining soil",
    "14":"Large catchments with few dams"
}
