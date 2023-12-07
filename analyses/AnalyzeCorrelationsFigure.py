import os
import pandas as pd
from data.metadata import *
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np

def modelScoresFigure(tag):

    linearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsLinear_" + str(tag) + ".csv")
    nonlinearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsNonlinear_" + str(tag) + ".csv")

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
    plt.ylim(-1.1,1.1)
    plt.title("Model Skill Levels")
    plt.ylabel("coefficient of determination")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(figurePath, "regressionScores_" + str(tag) + ".png"))#"linearRegressionScores.png"))
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
        if val == "score":
            groups.append("score")
        else:
            groups.append(predictorsToCategory[val])
    df["group"]  = groups
    df = df.groupby("group").sum()

    scoreMask = df.index == "score"
    scores = 100 * np.array(df[scoreMask])
    df = df[~scoreMask]
    df = (scores * df) / df.sum() # normalize to sum to 100 * coefficient of determination
    df = df.transpose()

    targetNames = list(df.index)

    return df, scores, targetNames

def barChartLinear(tag):
    linearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsLinear_" + str(tag) + ".csv")

    linearDf = pd.read_csv(linearDataPath)

    #linearDf = linearDf.drop("score", axis=1)
    for col in linearDf.columns[2:]:
        linearDf[col] = linearDf[col].abs()

    newTargets = []
    for target in linearDf["target"]:
        newTargets.append(predictablesToPretty[target])
    linearDf["target"] = newTargets

    linearDf = linearDf.groupby("target").mean()
    linearDf.to_csv(os.path.join(outputFilesPath, "feature_importances_linear_" + str(tag) + ".csv"), index=False) # save the results

    linearDf = linearDf.dropna(axis=1)
    linearDf, scores, targetNames = groupColumns(linearDf)
    ax = linearDf.plot(kind="bar",stacked=True, figsize=(12.5,5), edgecolor="black")
    plt.ylabel("% contribution")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title="feature category", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=0)
    plt.title("Contributions of Features to Linear Model Decision Making")
    plt.ylim(-2,np.max(linearDf.transpose().sum()) + 2)    
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "modelFeatureImportancesLinear_" + str(tag) + ".png"))
    plt.clf()

def barChartNonlinear(tag):
    nonlinearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsNonlinear_" + str(tag) + ".csv")

    nonlinearDf = pd.read_csv(nonlinearDataPath)

    #nonlinearDf = nonlinearDf.drop("score", axis=1)
    for col in nonlinearDf.columns[2:]:
        nonlinearDf[col] = nonlinearDf[col].abs()
    
    newTargets = []
    for target in nonlinearDf["target"]:
        newTargets.append(predictablesToPretty[target])
    nonlinearDf["target"] = newTargets

    nonlinearDf = nonlinearDf.groupby("target").mean()

    nonlinearDf.to_csv(os.path.join(outputFilesPath, "feature_importances_nonlinear_" + str(tag) + ".csv"), index=False) # save the results

    nonlinearDf = nonlinearDf.dropna(axis=1)
    nonlinearDf, scores, targetNames = groupColumns(nonlinearDf)
    ax = nonlinearDf.plot(kind="bar",stacked=True, figsize=(12.5,5), edgecolor="black")
    plt.ylabel("% contribution")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title="feature category", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=0)
    plt.title("Contributions of Features to Random Forest Model Decision Making")
    plt.xlabel("")
    plt.ylim(-2,np.max(nonlinearDf.transpose().sum()) + 2)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "modelFeatureImportancesNonlinear_" + str(tag) + ".png"))
    plt.clf()
    
    return scores, targetNames


def analyzeCorrelationsFigure():
    tags = ["raw","imputed"]
    
    tagScores = []
    for tag in tags:
        modelScoresFigure(tag)
        barChartLinear(tag)
        scores, labels = barChartNonlinear(tag)
        tagScores.append(np.squeeze(scores))
    improvements = tagScores[1] - tagScores[0]


    with open(os.path.join(logPath, "log_differencesPrePostImputing.txt"), "w+") as logFile:
        logFile.writelines("Changes in mean model accuracy following data imputation:\n")
        for i in range(len(labels)):
            logFile.writelines(str(labels[i].replace("\n"," ")) + ": ")
            logFile.writelines(str(improvements[i] / 100.) + "\n")
