import os
import pandas as pd
from data.metadata import *
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np

def modelScoresFigure():

    linearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsLinear_imputedPCA.csv")
    nonlinearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsNonlinear_imputedPCA.csv")

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
    plt.ylim(-1.1, 1.1)
    plt.title("Model Skill Levels")
    plt.ylabel("coefficient of determination")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(figurePath, "regressionScores_imputedPCA.png"))#"linearRegressionScores.png"))
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
            groups.append(predictorsToCategoryPCA[val])
    df["group"]  = groups
    df = df.groupby("group").sum()
    scoreMask = df.index == "score"
    scores = 100 * np.array(df[scoreMask])
    df = df[~scoreMask]
    df = (scores * df) / df.sum() # normalize to sum to 100 * coefficient of determination
    df = df.transpose()

    return df

def barChartLinear():
    linearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsLinear_imputedPCA.csv")
    linearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsLinear_imputedPCA.csv")

    linearDf = pd.read_csv(linearDataPath)

    #linearDf = linearDf.drop("score", axis=1)
    for col in linearDf.columns[2:]:
        linearDf[col] = linearDf[col].abs()

    newTargets = []
    for target in linearDf["target"]:
        newTargets.append(predictablesToPretty[target])
    linearDf["target"] = newTargets

    linearDf = linearDf.groupby("target").mean()
    linearDf.to_csv(os.path.join(outputFilesPath, "feature_importances_linear_imputedPCA.csv"), index=False) # save the results

    linearDf = linearDf.dropna(axis=1)
    linearDf = groupColumns(linearDf)
    ax = linearDf.plot(kind="bar",stacked=True, figsize=(15.5,5), edgecolor="black")
    plt.ylabel("% contribution")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title="feature category", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=0)
    plt.title("Contributions of Features to Linear Model Decision Making")
    plt.xlabel("")
    plt.ylim(-2,np.max(linearDf.transpose().sum()) + 2)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "modelFeatureImportancesLinear_imputedPCA.png"))
    plt.clf()

def barChartNonlinear():
    nonlinearDataPath = os.path.join(outputFilesPath, "regressionCoefficientsNonlinear_ImputedPCA.csv")

    nonlinearDf = pd.read_csv(nonlinearDataPath)

    #nonlinearDf = nonlinearDf.drop("score", axis=1)
    for col in nonlinearDf.columns[2:]:
        nonlinearDf[col] = nonlinearDf[col].abs()

    newTargets = []
    for target in nonlinearDf["target"]:
        newTargets.append(predictablesToPretty[target])
    nonlinearDf["target"] = newTargets

    nonlinearDf = nonlinearDf.groupby("target").mean()

    nonlinearDf.to_csv(os.path.join(outputFilesPath, "feature_importances_nonlinear_imputedPCA.csv"), index=False) # save the results

    nonlinearDf = nonlinearDf.dropna(axis=1)
    nonlinearDf = groupColumns(nonlinearDf)
    ax = nonlinearDf.plot(kind="bar",stacked=True, figsize=(15.5,5), edgecolor="black")
    plt.ylabel("% contribution")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title="feature category", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=0)
    plt.title("Contributions of Features to Random Forest Model Decision Making")
    plt.xlabel("")
    plt.ylim(-2,np.max(nonlinearDf.transpose().sum()) + 2)
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "modelFeatureImportancesNonlinear_imputedPCA.png"))
    plt.clf()



def analyzeCorrelationsFigurePCA():

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


