import os
import pandas as pd
from data.metadata import *
from analyses.colorCatchments import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn



def computeM():
    ''' 
    Solve for the value of "m" in Fuh's Equation
    '''

    dataFilePath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df = pd.read_csv(dataFilePath)
 
    ws = torch.nn.Parameter(torch.rand(len(df["catchment"])) + 1) # initialize
    y = torch.Tensor(df["d_pMean"])
    p_pets = torch.Tensor(df["p_petMean"])
    optimizer = optim.Adam([ws], lr=1e-3)
    numEpochs = 10000
    loop = tqdm(total=numEpochs)
    
    print("solving for m from Fuh's equation")
    losses = []
    for epoch in range(numEpochs):
        optimizer.zero_grad()
        yHat = torch.pow(1 + torch.pow(p_pets, -1 * ws), 1. / ws)  - torch.pow(p_pets, -1)
        loss = torch.linalg.norm(yHat - y)
        loss.backward()
        optimizer.step()
        loop.set_description("loss: " + str(loss.detach().item()))
        loop.update(1)
        losses.append(loss.detach().item())
    loop.close()

    plt.plot(losses)
    plt.title("Losses for Solving for M")
    plt.ylabel("|| q/p$^*$ - q/p ||$_2$")
    plt.xlabel("iteration")
    plt.savefig(os.path.join(figurePath, "loss_solving_for_m.png"))
    
    df["m"] = ws.detach().numpy()
    outputPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_imputed.csv")
    df.to_csv(outputPath, index=False)

    rawPath = os.path.join(outputFilesPath, "combinedTimeseriesSummariesAndMetadata_raw.csv")
    rdf = pd.read_csv(rawPath)
    rdf["m"] = ws.detach().numpy()
    rdf.to_csv(rawPath, index=False)

