import os
import pandas as pd
from data.metadata import *
from analyses.colorCatchments import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from colorCatchments import getColors
from scipy.stats import spearmanr
import copy


def analyzeSpearmanCorrelations():
    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(dataFilePath)
    df = df[~df["pommfSlope"].isna()]

    absos = [True, False]

    for abso in absos:
        for predictable in list(predictablesToPretty.keys()) + ["percent_deficit", "budget_deficit"]:
            ldf = copy.copy(df[[predictable] + list(predictorsToPretty.keys())])
            ldf = ldf.dropna()
            cols = np.array(ldf.columns)
            if abso:
                ldf[predictable] = ldf[predictable].abs()
            result = spearmanr(ldf.to_numpy())
            correlations = result[0][0,:]
            pvals = result[1][0,:]
            indices = np.flip(np.argsort(np.abs(correlations)))

            if abso:
                outDf = pd.DataFrame.from_dict({"absolute_predictors":cols[indices[1:]],"correlations":correlations[indices[1:]],"pvals":pvals[indices[1:]]})
                outDf.to_csv(os.path.join(outputFilesPath, "individualCorrs_abs_" + str(predictable) + ".csv"), index=False)
            else:
                outDf = pd.DataFrame.from_dict({"predictors":cols[indices[1:]],"correlations":correlations[indices[1:]],"pvals":pvals[indices[1:]]})
                outDf.to_csv(os.path.join(outputFilesPath, "individualCorrs_" + str(predictable) + ".csv"), index=False)



