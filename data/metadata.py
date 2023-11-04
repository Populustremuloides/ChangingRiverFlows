import os
import sys
import numpy as np
np.random.seed(0)

# stored variable names for accessing the raw data

# global variables are labeled with g_*
g_numYearsToUseForAnalysis = 9

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

# get the outputFiles
outputFilesPath = os.path.join(outputPath, "outputFiles")
augmentedTimeseriesPath = os.path.join(outputPath, "augmentedTimeseries")
figurePath = os.path.join(outputPath, "figures")


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
    "p_petSlope":"yearly change in precipitaion / potential evapotranspiration",
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

    "cls1":"percent evergreen deciduous trees",
    "cls2":"percent braodleaf evergree trees",
    "cls3":"percent deciduous broadleaf trees",
    "cls4":"percent mixed other trees",
    "cls5":"percent shrubs",
    "cls6":"percent herbacious vegetation",
    "cls7":"percent cultivated and managed vegetation",
    "cls8":"percent regularly flooded vegetation",
    "cls9":"percent urbat",
    "cls10":"percent snow/ice",
    "cls11":"percent barren",
    "cls12":"percent open water",

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
    "p_petSlope":"yearly change in precipitaion / potential evapotranspiration",
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
    "p_petSlope":"climate ratio",
    "p_petPercentChange":"climate ratio",
    "dompSlope":"precipitation",
    "maspSlope":"precipitation",
    "maspPercentChange":"precipitation",
    "p_petMean":"climate ratio",
    "maspMean":"precipitation",
    "dompMean":"precipitation",
    "pet_pSlope":"climate ratio",
    "pet_pPercentChange":"climate ratio",
    "doppetMean":"potential evapotranspiration",
    "pet_etSlope":"climate ratio",
    "pet_etPercentChange":"climate ratio",
    "dompetSlope":"potential evapotranspiration",
    "maspetSlope":"potential evapotranspiration",
    "maspetPercentChange":"potential evapotranspiration",

    "et_pSlope":"climate ratio",
    "et_pPercentChange":"climate ratio",
    "dometSlope":"evapotranspiration",
    "masetSlope":"evapotranspiration",
    "masetPercentChange":"evapotranspiration",
    "pet_pMean":"climate ratio",
    "pet_etMean":"climate ratio",
    "dompetMean":"potential evapotranspiration",
    "maspetMean":"potential evapotranspiration",
    "d_pMean":"mean flow properties",
    "masdMean":"mean flow properties",
    "domfMean":"mean flow properties",
    "et_pMean":"climate ratio",
    "dometMean":"evapotranspiration",
    "masetMean":"evapotranspiration",
    "dopfMean":"mean flow properties",
    "doppMean":"precipitation",
    "doptMean":"temperature",
    "pommfMean":"mean flow properties",
    "pommpMean":"precipitation",
    "pommtSlope":"temperature",
    "doppetSlope":"potential evapotranspiration",
    "matSlope":"temperature",
    "domtSlope":"temperature",
    "doppSlope":"precipitation",
    "doptSlope":"temperature",
    "pommpetSlope":"potential evapotranspiration",
    "pommpSlope":"precipitation",
    "pommpetMean":"potential evapotranspiration",
    "dopetSlope":"evapotranspiration",
    "matMean":"temperature",
    "domtMean":"temperature",
    "pommetMean":"evapotranspiration",
    "dopetMean":"evapotranspiration",
    "pommtMean":"temperature",
    "pommetSlope":"evapotranspiration",

    "cls1":"land cover",
    "cls2":"land cover",
    "cls3":"land cover",
    "cls4":"land cover",
    "cls5":"land cover",
    "cls6":"land cover",
    "cls7":"direct human impact",
    "cls8":"land cover",
    "cls9":"direct human impact",
    "cls10":"land cover",
    "cls11":"land cover",
    "cls12":"land cover",

    "Dam_SurfaceArea":"dams",
    "Dam_Count":"dams",
    "MeanPopden_2000":"direct human impact",
    "MeanPopden_2005":"direct human impact",
    "MeanPopden_2010":"direct human impact",
    "MeanPopden_2015":"direct human impact",
    "MeanHumanFootprint":"direct human impact",

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
    "p_petSlope":"climate ratio",
    "p_petPercentChange":"climate ratio",
    "dompSlope":"precipitation",
    "maspSlope":"precipitation",
    "maspPercentChange":"precipitation",
    "p_petMean":"climate ratio",
    "maspMean":"precipitation",
    "dompMean":"precipitation",
    "pet_pSlope":"climate ratio",
    "pet_pPercentChange":"climate ratio",
    "doppetMean":"potential evapotranspiration",
    "pet_etSlope":"climate ratio",
    "pet_etPercentChange":"climate ratio",
    "dompetSlope":"potential evapotranspiration",
    "maspetSlope":"potential evapotranspiration",
    "maspetPercentChange":"potential evapotranspiration",

    "et_pSlope":"climate ratio",
    "et_pPercentChange":"climate ratio",
    "dometSlope":"evapotranspiration",
    "masetSlope":"evapotranspiration",
    "masetPercentChange":"evapotranspiration",
    "pet_pMean":"climate ratio",
    "pet_etMean":"climate ratio",
    "dompetMean":"potential evapotranspiration",
    "maspetMean":"potential evapotranspiration",
    "d_pMean":"mean flow properties",
    "masdMean":"mean flow properties",
    "domfMean":"mean flow properties",
    "et_pMean":"climate ratio",
    "dometMean":"evapotranspiration",
    "masetMean":"evapotranspiration",
    "dopfMean":"mean flow properties",
    "doppMean":"precipitation",
    "doptMean":"temperature",
    "pommfMean":"mean flow properties",
    "pommpMean":"precipitation",
    "pommtSlope":"temperature",
    "doppetSlope":"potential evapotranspiration",
    "matSlope":"temperature",
    "domtSlope":"temperature",
    "doppSlope":"precipitation",
    "doptSlope":"temperature",
    "pommpetSlope":"potential evapotranspiration",
    "pommpSlope":"precipitation",
    "pommpetMean":"potential evapotranspiration",
    "dopetSlope":"evapotranspiration",
    "matMean":"temperature",
    "domtMean":"temperature",
    "pommetMean":"evapotranspiration",
    "dopetMean":"evapotranspiration",
    "pommtMean":"temperature",
    "pommetSlope":"evapotranspiration",

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
