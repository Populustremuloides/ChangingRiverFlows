import os
import pandas as pd
from data.metadata import *
from analyses.utilityFunctions import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor

def getDroppers(df, tag):
    droppers = []
    keepers = []

    with open(os.path.join(logPath, "log_analysis_nonlinearCorelations_" + str(tag) + ".txt"), "w+") as logFile:
        logFile.writelines("The following columns were dropped from the analysis because they had > 0.1 proportion nan values:\n")
        for col in df.columns:
            proportionNan =  float(np.sum(df[col].isna())) / float(len(df[col]))
            if proportionNan > 0.1:
                droppers.append(col)
                print("dropping", col, "for linear regression analysis because of sparsity.")
                logFile.writelines(col + "\n")
            else:
                keepers.append(col)
    return droppers, keepers

def getTrainMask(df):
    # identify the number of test and train points to use
    numPoints = len(df[df.columns[0]])
    numTest = int(numPoints * 0.2)
    numTrain = int(numPoints * 0.8)
    diff =  (numTest + numTrain) - numPoints
    numTrain = numTrain - diff

    mask = np.array(([False] * numTest) + ([True] * numTrain))
    return mask

# cycle through the data and calculate the runoff ratios
def nonlinearAnalysis(df, tag, numRepeats):

    predictableVars = list(predictablesToPretty.keys())
    predictorVars = list(predictorsToPretty.keys())

    predictorsDf = df[predictorVars]
    droppers, keepers  = getDroppers(predictorsDf, tag)

    # prepare to save the data
    dataDict = {"target":[],"score":[]}
    for predictorVar in predictorVars:
        dataDict[predictorVar] = []

    loop = tqdm(total=numRepeats * len(predictableVars))
    for i in range(numRepeats):
        for predictable in predictableVars:
            # drop nan values
            ldf = copy.copy(df[keepers + [predictable]])
            ldf = ldf.dropna()

            # normalize
            ldf = ldf - ldf.mean()
            ldf = ldf / ldf.std()

            # separate test vs train
            mask = getTrainMask(ldf)
            mask = np.random.choice(mask, size=mask.shape[0], replace=False)
            trainDf = ldf[mask]
            testDf = ldf[~mask]

            # separate out target vs features
            yTest = testDf[predictable].to_numpy()
            xTest = testDf[testDf.columns[:-1]].to_numpy()

            yTrain = trainDf[predictable].to_numpy()
            xTrain = trainDf[trainDf.columns[:-1]].to_numpy()

            # run the model
            model = RandomForestRegressor()
            model.fit(xTrain,yTrain)
            score = model.score(xTest, yTest)
            importances = model.feature_importances_

            cols = list(trainDf.columns[:-1])
            colsToImportances = dict(zip(cols, importances))

            # save the data
            dataDict["target"].append(predictable)
            dataDict["score"].append(score)

            for predictorVar in predictorVars:
                if predictorVar in colsToImportances.keys():
                    dataDict[predictorVar].append(colsToImportances[predictorVar])
                else:
                    dataDict[predictorVar].append(None)

            loop.set_description("Computing nonlinear correlates of changes in flow")
            loop.update(1)

    loop.close()

    coefficientsDf = pd.DataFrame.from_dict(dataDict)
    coefficientsDf.to_csv(os.path.join(outputFilesPath, "regressionCoefficientsNonlinear_" + str(tag) + ".csv"), index=False)


def analyzeCorrelationsNonlinear():
    numRepeats = 10

    tags = ["raw","imputed"]

    for tag in tags:
        dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_" + str(tag) + ".csv")
        df = pd.read_csv(dataFilePath)
        df = df[np.array(~df["p_petSlope"].isna())] # keep only the rows for which we have data
        nonlinearAnalysis(df, tag, numRepeats)

