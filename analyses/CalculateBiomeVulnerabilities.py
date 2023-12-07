import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def renameBiome(df):
    ''' rename the biome column for easier viewing in the plots '''
    newBiome = []
    biome = df["BIOME"]
    for element in biome:
        if element == "Tropical and subtropical grasslands, savannas, and shrublands":
            newBiome.append("Tropical and Subtropical Grasslands,\nSavannas, and Shrublands")
        elif element == "Temperate Grasslands, Savannas, and Shrublands":
            newBiome.append("Temperate Grasslands,\nSavannas, and Shrublands")
        elif element == "Tropical and subtropical grasslands, savannas, and shrublands":
            newBiome.append("Tropical and subtropical grasslands,\nsavannas, and shrublands")
        elif element == "Mediterranean Forests, Woodlands, and Scrub":
            newBiome.append("Mediterranean Forests,\nWoodlands, and Scrub")
        elif element == "Tropical and Subtropical Moist Broadleaf Forests":
            newBiome.append("Tropical and Subtropical\nMoist Broadleaf Forests")
        elif element == "Tropical and Subtropical Dry Broadleaf Forests":
            newBiome.append("Tropical and Subtropical\nDry Broadleaf Forests")
        else:
            newBiome.append(element)

    df["BIOME"] = newBiome
    return df

def plotVar(outDf, ldf, var, saveTitle):
    # Sort the BIOME categories based on the magnitude
    magnitude = ldf.groupby("BIOME")[var].mean()
    sortedCategories = magnitude.sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.boxplot(data=outDf, x="BIOME", y=var, hue="data source", ax=ax, order=sortedCategories)
    plt.xticks(rotation=90, fontsize=15)
    #plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.grid(axis="y")
    plt.ylabel(var, fontsize=15)
    plt.xlabel("")
    plt.savefig(os.path.join(figurePath, saveTitle))
    plt.clf()
    plt.close()


def calculateBiomeVulnerabilities():
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    df = pd.read_csv(dataFilePath)
    biomeMask = np.array(~df["BIOME"].isna())
    df = df[biomeMask]
    pommfMask = np.array(~df["pommfSlope"].isna())
    df = df[pommfMask]
    df = renameBiome(df)
    print(df)

    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedRF.csv")
    dfAll = pd.read_csv(dataFilePath)
    dfAll = dfAll[biomeMask]
    dfAll = dfAll[~pommfMask]
    dfAll = dfAll[~dfAll["pommfSlope"].isna()] 
    dfAll = renameBiome(dfAll)

    dataDict = {"data source":[],"BIOME":[],"Yearly Absolute % Change in Runoff Ratio":[], "Yearly Absolute % Change in Mean Annual Specific Discharge":[], "Yearly Absolute Change in Mean Annual Specific Discharge (L / km$^2$)":[], "Yearly Absolute Change in Day of Mean Flow (days)":[], "Yearly Absolute Change in Day of Peak Flow (days)":[], "Yearly Absolute Change in Period of Mean Flow (days)":[], "Yearly % Change in Runoff Ratio":[], "Yearly % Change in Mean Annual Specific Discharge":[], "Yearly Change in Mean Annual Specific Discharge (L / km$^2$)":[], "Yearly Change in Day of Mean Flow (days)":[], "Yearly Change in Day of Peak Flow (days)":[], "Yearly Change in Period of Mean Flow (days)":[], "Absolute % Budget Deficit":[], "% Budget Deficit":[]}
    
    for index, row in df.iterrows():
        dataDict["data source"].append("Measured")
        dataDict["BIOME"].append(row["BIOME"])
        dataDict["Yearly Absolute % Change in Runoff Ratio"].append(abs(row["d_pPercentChange"]))
        dataDict["Yearly Absolute % Change in Mean Annual Specific Discharge"].append(abs(row["masdPercentChange"]))
        dataDict["Yearly Absolute Change in Mean Annual Specific Discharge (L / km$^2$)"].append(abs(row["masdSlope"]))
        dataDict["Yearly Absolute Change in Day of Mean Flow (days)"].append(abs(row["domfSlope"]))
        dataDict["Yearly Absolute Change in Day of Peak Flow (days)"].append(abs(row["dopfSlope"]))
        dataDict["Yearly Absolute Change in Period of Mean Flow (days)"].append(abs(row["pommfSlope"]))
        dataDict["Absolute % Budget Deficit"].append(abs(row["percent_deficit"]))

        dataDict["Yearly % Change in Runoff Ratio"].append(row["d_pPercentChange"])
        dataDict["Yearly % Change in Mean Annual Specific Discharge"].append(row["masdPercentChange"])
        dataDict["Yearly Change in Mean Annual Specific Discharge (L / km$^2$)"].append(row["masdSlope"])
        dataDict["Yearly Change in Day of Mean Flow (days)"].append(row["domfSlope"])
        dataDict["Yearly Change in Day of Peak Flow (days)"].append(row["dopfSlope"])
        dataDict["Yearly Change in Period of Mean Flow (days)"].append(row["pommfSlope"])
        dataDict["% Budget Deficit"].append(row["percent_deficit"])

    for index, row in dfAll.iterrows():
        dataDict["data source"].append("Imputed")
        dataDict["BIOME"].append(row["BIOME"])
        dataDict["Yearly Absolute % Change in Runoff Ratio"].append(abs(row["d_pPercentChange"]))
        dataDict["Yearly Absolute % Change in Mean Annual Specific Discharge"].append(abs(row["masdPercentChange"]))
        dataDict["Yearly Absolute Change in Mean Annual Specific Discharge (L / km$^2$)"].append(abs(row["masdSlope"]))
        dataDict["Yearly Absolute Change in Day of Mean Flow (days)"].append(abs(row["domfSlope"]))
        dataDict["Yearly Absolute Change in Day of Peak Flow (days)"].append(abs(row["dopfSlope"]))
        dataDict["Yearly Absolute Change in Period of Mean Flow (days)"].append(abs(row["pommfSlope"]))
        dataDict["Absolute % Budget Deficit"].append(abs(row["percent_deficit"]))

        dataDict["Yearly % Change in Runoff Ratio"].append(row["d_pPercentChange"])
        dataDict["Yearly % Change in Mean Annual Specific Discharge"].append(row["masdPercentChange"])
        dataDict["Yearly Change in Mean Annual Specific Discharge (L / km$^2$)"].append(row["masdSlope"])
        dataDict["Yearly Change in Day of Mean Flow (days)"].append(row["domfSlope"])
        dataDict["Yearly Change in Day of Peak Flow (days)"].append(row["dopfSlope"])
        dataDict["Yearly Change in Period of Mean Flow (days)"].append(row["pommfSlope"])
        dataDict["% Budget Deficit"].append(row["percent_deficit"])

    outDf = pd.DataFrame.from_dict(dataDict)
    outDf = outDf.dropna() # double check no NaNs got in there
    
    print("before outlier")
    print(outDf)
    outDf = outDf[outDf["Yearly Change in Day of Mean Flow (days)"] < 15] # remove one outlier
    print(outDf)
    print("after outlier")

    ldf = outDf[outDf["data source"] == "Measured"]

    plotVar(outDf, ldf, "% Budget Deficit", "boxplot_budgetDeficit.png")
    plotVar(outDf, ldf, "Absolute % Budget Deficit", "boxplot_absbudgetDeficit.png")

    plotVar(outDf, ldf, "Yearly Absolute % Change in Runoff Ratio", "boxplot_absRunoffRatioSlope.png")
    plotVar(outDf, ldf, "Yearly Absolute Change in Mean Annual Specific Discharge (L / km$^2$)", "boxplot_absMASDSlope.png")
    plotVar(outDf, ldf, "Yearly Absolute % Change in Mean Annual Specific Discharge", "boxplot_absMASDPercentChange.png")
    plotVar(outDf, ldf, "Yearly Absolute Change in Day of Mean Flow (days)", "boxplot_absDOMFSlope.png")
    plotVar(outDf, ldf, "Yearly Absolute Change in Period of Mean Flow (days)", "boxplot_absPOMMFSlope.png")

    plotVar(outDf, ldf, "Yearly % Change in Runoff Ratio", "boxplot_RunoffRatioSlope.png")
    plotVar(outDf, ldf, "Yearly Change in Mean Annual Specific Discharge (L / km$^2$)", "boxplot_MASDSlope.png")
    plotVar(outDf, ldf, "Yearly % Change in Mean Annual Specific Discharge", "boxplot_MASDPercentChange.png")
    plotVar(outDf, ldf, "Yearly Change in Day of Mean Flow (days)", "boxplot_DOMFSlope.png")
    plotVar(outDf, ldf, "Yearly Change in Period of Mean Flow (days)", "boxplot_POMMFSlope.png")
