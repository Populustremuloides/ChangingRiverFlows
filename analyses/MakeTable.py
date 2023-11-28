import pandas as pd
from data.metadata import *

def makeTable(numToShow=5):
    dataDict = {}
    for predictable in list(predictablesToPretty.keys()):
        colName = predictablesToPretty[predictable]
        colName = colName.replace("\n"," ")
        dataDict[colName] = []
        ldf = pd.read_csv(os.path.join(outputFilesPath, "individualCorrs_" + str(predictable) + ".csv"))
        for index, row in ldf.iterrows():
            if index > numToShow:
                break
            dataDict[colName].append(predictorsToPretty[row["predictors"]] + " ({:.2f}".format(row["correlations"]) + ")")
    outDf = pd.DataFrame.from_dict(dataDict)
    outDf.to_csv(os.path.join(outputFilesPath, "topSpearmanCorrelates.csv"), index=False)
