import pandas as pd
from data.metadata import *
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import copy

def pcaMetadata():


    path = os.path.join(outputFilesPath,"combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(path)

    catchments = np.array(df["catchment"])
    predictablesDf = copy.copy(df[list(predictablesToPretty.keys()) + ["catchment"]])

    # remove ordinal variables
    droppers = ["River", "Station", "ID", "FEOW_ID", "Country","LINKNO","Ecoregion_Name","Continent","BIOME","ECO_NAME", "quality"]
    df = df.drop(droppers + ["catchment"], axis=1)
    df = df.drop(list(predictablesToPretty.keys()), axis=1)
    cols = df.columns
    
    # normalize
    df = df - df.mean()
    df = df / df.std()

    # run pca
    data = df.to_numpy()
    model = PCA(svd_solver="full")
    model.fit(data)
    transformedData = model.transform(data)

    componentIndex = np.arange(len(model.explained_variance_ratio_)) + 1 # an index list for plotting

    # plot how the contributed variance explained drops off
    plt.scatter(componentIndex, model.explained_variance_ratio_)
    plt.xlabel("principal component number")
    plt.title("Linear Structure in Prediction Features")
    plt.ylabel("additional proportion variance explained")
    plt.savefig(os.path.join(figurePath, "pca_components.png"))
    plt.clf()


    outDf = pd.DataFrame(transformedData[:,:g_numPCAVarsToKeep], columns=np.arange(g_numPCAVarsToKeep) + 1)
    outDf["catchment"] = catchments # add back in the features that will be predicted later on
    outDf = predictablesDf.merge(outDf, on="catchment")
    print(outDf)
    
    # log what we did
    with open(os.path.join(logPath, "log_PCA_analysis.txt"), "w+") as logFile:
        logFile.writelines("The following columns were dropped from the PCA analysis because they were ordinal values:\n")
        for dropped in droppers:
            logFile.writelines(dropped + "\n")
        logFile.writelines("\n\n")
        logFile.writelines("proportion variance explained of kept features: " + str(np.sum(model.explained_variance_ratio_[:g_numPCAVarsToKeep])) + "\n")
        logFile.writelines("proportion variance explained of individual features:\n")
        for i, pve in enumerate(model.explained_variance_ratio_[:g_numPCAVarsToKeep]):
            logFile.writelines("feature " + str(pve) + "\n")
    
    # save the transformed data
    path = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputedPCA.csv")
    outDf.to_csv(path, index=False)

    # save the feature weights for each component
    for i in np.arange(g_numPCAVarsToKeep):
        weights = model.components_[i]
        indices = np.flip(np.argsort(np.abs(weights)))
        weightsDf = pd.DataFrame.from_dict({"weights":weights[indices],"labels":cols[indices]})
        weightsDf.to_csv(os.path.join(outputFilesPath, "pca_feature_importances_component_" + str(i + 1) + ".csv"), index=False)


