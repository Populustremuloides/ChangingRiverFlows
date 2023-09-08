
import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# cycle through the data and calculate the runoff ratios

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
    "d_pPercentChange":"yearly % change in runoff ratio",
    "d_pSlope":"yearly change in runoff ratio",
    "domfSlope":"yearly change in day of mean flow",
    "masdSlope":"yearly change in mean annual specific discharge",
    "masdPercentChange":"yearly % change in mean annual specific discharge",
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
    "dopfSlope":"yearly change in day of peak flow",
    "matSlope":"yearly change in mean annual temperature",
    "domtSlope":"yearly change in day of mean temperature",
    "doppSlope":"yearly change in day of peak precipitation",
    "doptSlope":"yearly change in day of peak temperature",
    "pommpetSlope":"yearly change in period of mean potential evapotranspiration",
    "pommpSlope":"yearly change in day of mean precipitation",
    "pommpetMean":"mean value of day of mean potential evapotranspiration",
    "dopetSlope":"day of peak evapotranspiration",
    "pommfSlope":"yearly change in period of mean flow",
    "matMean":"mean value of mean annual temperature",
    "dommtMean":"mean value of day of mean temperature",
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
    "gelev_m":"elevation",
    "drain_den":"drainage density",

}

def analyzeTempAndPrecip():
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata.csv")
    df = pd.read_csv(dataFilePath)

    df[""]




