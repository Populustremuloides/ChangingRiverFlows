import torch
import torch.optim as optim
import pandas as pd
from data.metadata import *
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# read in the data

def imputeMetadata():
    '''
    use gradient descent to impute data in a way that
    does not alter the correlation structure of the
    variables with each other
    '''

    numIterations = 10000

    # read in the metadata file
    df = pd.read_csv(os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv"))

    #df = df.drop(df.columns[0], axis=1)

    # identify the columns we want to use
    keeperCols = []
    for col in df.columns:
        try:
            float(df[col][1])
            if col != "LINKNO" and col != "catchment" and col != "ID" and col != "FEOW_ID":
                keeperCols.append(col)
        except:
            pass


    # identify missing data
    ddf = df[keeperCols]
    
    # save this values for later
    means = ddf.mean()
    stds = ddf.std()


    ddf = ddf - means #ddf.mean()
    ddf = ddf / stds #ddf.std()
    data = ddf.to_numpy().T
    data = np.array(data, dtype=np.float32)
    mask = torch.from_numpy(np.isnan(data))
    data = torch.from_numpy(data)

    targetCorrelations = torch.tensor(ddf.corr().to_numpy())
    targetCovariances = torch.tensor(ddf.cov().to_numpy())

    # initialize imputed data
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer.fit(data)
    imputedData = torch.nn.Parameter(torch.Tensor(imputer.transform(data)))

    optimizer = optim.Adam([imputedData], lr=1e-3)
    dataToAnalyze = data

    losses = []
    loop = tqdm(total=numIterations)
    loop.set_description("mapping catchments")
    for i in range(numIterations):
        optimizer.zero_grad()

        dataToAnalyzeCopy = dataToAnalyze.clone() # copy the original data
        dataToAnalyzeCopy[mask] = imputedData[mask] # fill in the imputed data

        # measure differences in statistics between the unimputed data
        # and the imputed data
        imputedCorr = torch.corrcoef(dataToAnalyzeCopy)
        diffCorr = imputedCorr - targetCorrelations
        imputedCov = torch.cov(dataToAnalyzeCopy)
        diffCov = imputedCov - targetCovariances

        # calculate the loss / update imputed data
        loss = torch.sum(torch.abs(diffCorr.flatten())) + torch.sum(torch.abs(diffCov.flatten()))
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()

        loop.update(1)
        loop.set_description("imputing data loss: " + str(loss.detach().numpy()) + " ")

    # visualize the correlation matrices before and after

    with torch.no_grad():
        dataToAnalyzeCopy = dataToAnalyze.clone() # copy the original data
        dataToAnalyzeCopy[mask] = imputedData[mask] # fill in the imputed data

        imputedCorr = torch.corrcoef(dataToAnalyzeCopy)

        fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(12,5))
        sns.heatmap(targetCorrelations, ax=ax[0], center=0, vmin=-1.1, vmax=1.1)
        sns.heatmap(imputedCorr, ax=ax[1], center=0, vmin=-1.1, vmax=1.1)
        sns.heatmap(targetCorrelations - imputedCorr, ax=ax[2], center=0, vmin=-1.1, vmax=1.1)
        ax[0].set_title("Raw Correlations")
        ax[1].set_title("Imputed Correlations")
        ax[2].set_title("(Raw - Imputed) Correlations")
        ax[0].set_xlabel("Feature Index")
        ax[0].set_ylabel("Feature Index")
        ax[1].set_xlabel("Feature Index")
        ax[2].set_xlabel("Feature Index")
        plt.tight_layout()
        plt.savefig(os.path.join(figurePath, "imputedVsRawCorrelations.png"))
        plt.clf()

        imputedCov = torch.cov(dataToAnalyzeCopy)
        fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(12,5))
        sns.heatmap(targetCovariances, ax=ax[0], center=0, vmin=-1.1, vmax=1.1)
        sns.heatmap(imputedCov, ax=ax[1], center=0, vmin=-1.1, vmax=1.1)
        sns.heatmap(targetCovariances- imputedCov, ax=ax[2], center=0, vmin=-1.1, vmax=1.1)
        ax[0].set_title("Raw Covariances")
        ax[1].set_title("Imputed Covariances")
        ax[2].set_title("(Raw - Imputed) Covariances")
        ax[0].set_xlabel("Feature Index")
        ax[0].set_ylabel("Feature Index")
        ax[1].set_xlabel("Feature Index")
        ax[2].set_xlabel("Feature Index")
        plt.tight_layout()
        plt.savefig(os.path.join(figurePath, "imputedVsRawCovariances.png"))
        plt.clf()


    # plot the loss through updates
    fig, ax = plt.subplots(figsize=(5,7))
    ax.plot(losses)
    ax.set_xlabel("Number of Updates")
    ax.set_ylabel(r"$\sum_i \sum_j |Cov_{ij}(Imputed) - Cov_{ij}(Raw)| + \sum_i \sum_j |Corr_{ij}(Imputed) - Corr_{ij}(Raw)|$")
    ax.set_title("Imputation Loss During Optimization")
    plt.tight_layout()
    plt.savefig(os.path.join(figurePath, "imputationOptimizationLoss.png"))
    plt.clf()

    # save a log of how the imputation process went
    logerPath = os.path.join(logPath, "log_imputingLog.txt")
    with open(logerPath, "w+") as logFile:
        logFile.write("Loss = " + r"$\sum_i \sum_j |Cov_{ij}(Imputed) - Cov_{ij}(Raw)| + \sum_i \sum_j |Corr_{ij}(Imputed) - Corr_{ij}(Raw)|$")
        logFile.write("data were transformed to have mean zero and standard deviation of 1 prior to imputation.")
        logFile.write("imputed values were initialized to the distribution mean.")
        logFile.write("final loss: " + str(losses[-1]))
    
    # save the data
    with torch.no_grad():
        dataToAnalyzeCopy = dataToAnalyze.clone() # copy the original data
        dataToAnalyzeCopy[mask] = imputedData[mask]
        dataToAnalyzeCopy.numpy()
        outDf = pd.DataFrame(dataToAnalyzeCopy.T, columns=keeperCols, index=list(range(len(list(df["catchment"])))))
        #outDf["catchment"] = df["catchment"]

        # re-scale to have the same values as the original data
        outDf = (outDf * stds) + means

        # replace the original data
        for col in outDf.columns:
            df[col] = outDf[col]

        # nobody will ever know the difference :) (except that they have different names)
        path = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
        df.to_csv(path, index=False)

