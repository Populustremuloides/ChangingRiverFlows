import os
import pandas as pd
from data.metadata import *
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np

def modelScoresFigure():
    linearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsLinear.csv")
    nonlinearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsNonlinear.csv")

    linearDf = pd.read_csv(linearDataPath)
    nonlinearDf = pd.read_csv(nonlinearDataPath)

    # plot the scores *************************************

    newTargets = []
    for target in linearDf["target"]:
        newTargets.append(predictablesToPretty[target])
    linearDf["target"] = newTargets

    newTargets = []
    for target in nonlinearDf["target"]:
        newTargets.append(predictablesToPretty[target])
    nonlinearDf["target"] = newTargets

    ldf = copy.copy(linearDf[["score", "target"]])
    ldf["type"] = ["linear"] * len(ldf["score"])

    ndf = copy.copy(nonlinearDf[["score", "target"]])
    ndf["type"] = ["nonlinear"] * len(ndf["score"])

    scoreDf = pd.concat([ldf, ndf])

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(ax=ax, data=scoreDf, x="target",y="score", hue="type")
    plt.title("Linear Model Skill Levels")
    plt.ylabel("coefficient of determination")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(figurePath, "regressionScores.png"))#"linearRegressionScores.png"))
    plt.clf()

    #fig, ax = plt.subplots(figsize=(11, 5))
    #sns.violinplot(ax=ax, data=ndf, x="target",y="score")
    #plt.title("Random Forest Model Skill Levels")
    #plt.ylabel("coefficient of determination")
    #plt.xticks(rotation=0)
    #plt.savefig(os.path.join(figurePath, "nonlinearRegressionScores.png"))
    #plt.show()

def groupColumns(df):
    df = df.transpose()

    groups = []
    for val in df.index:
        groups.append(predictorsToCategory[val])
    df["group"]  = groups
    df = df.groupby("group").sum()
    df = 100 * df / df.sum()
    df = df.transpose()

    return df

def barChartLinear():
    linearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsLinear.csv")

    linearDf = pd.read_csv(linearDataPath)

    linearDf = linearDf.drop("score", axis=1)
    for col in linearDf.columns[1:]:
        linearDf[col] = linearDf[col].abs()

    newTargets = []
    for target in linearDf["target"]:
        newTargets.append(predictablesToPretty[target])
    linearDf["target"] = newTargets

    linearDf = linearDf.groupby("target").mean()
    linearDf.to_csv(os.path.join(outputFilesPath, "feature_importances_linear.csv"), index=False) # save the results

    linearDf = linearDf.dropna(axis=1)
    linearDf = groupColumns(linearDf)
    ax = linearDf.plot(kind="bar",stacked=True, figsize=(12.5,5), edgecolor="black")
    plt.ylabel("% contribution")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title="feature category", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=0)
    plt.title("Contributions of Features to Linear Model Decision Making")
    plt.ylim(-2,102)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "modelFeatureImportancesLinear.png"))
    plt.clf()

def barChartNonlinear():
    nonlinearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsNonlinear.csv")

    nonlinearDf = pd.read_csv(nonlinearDataPath)

    nonlinearDf = nonlinearDf.drop("score", axis=1)
    for col in nonlinearDf.columns[1:]:
        nonlinearDf[col] = nonlinearDf[col].abs()

    newTargets = []
    for target in nonlinearDf["target"]:
        newTargets.append(predictablesToPretty[target])
    nonlinearDf["target"] = newTargets

    nonlinearDf = nonlinearDf.groupby("target").mean()
    print(nonlinearDf)
    quit()
    nonlinearDf.to_csv(os.path.join(outputFilesPath, "feature_importances_nonlinear.csv"), index=False) # save the results

    nonlinearDf = nonlinearDf.dropna(axis=1)
    nonlinearDf = groupColumns(nonlinearDf)
    ax = nonlinearDf.plot(kind="bar",stacked=True, figsize=(12.5,5), edgecolor="black")
    plt.ylabel("% contribution")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title="feature category", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=0)
    plt.title("Contributions of Features to Random Forest Model Decision Making")
    plt.ylim(-2,102)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "modelFeatureImportancesNonlinear.png"))
    plt.clf()



def analyzeCorrelationsFigure():
    modelScoresFigure()
    barChartLinear()
    barChartNonlinear()




    # make the index be the treatment type *****************************************************



    #print(linearDf)
    #print(nonlinearDf)



    # group into parts

    # absoltue value
    # mean values
    # normalize

    # create df

    # plot


