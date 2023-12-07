from analyses.CombineTimeseries import combineTimeseries
from analyses.AddLocalWaterYear import addLocalWaterYear
from analyses.CalculateBudgetDeficits import calculateBudgetDeficits

# ratio analyses
from analyses.AnalyzeD_P import analyzeD_P
from analyses.AnalyzePET_ET import analyzePET_ET
from analyses.AnalyzePET_P import analyzePET_P
from analyses.AnalyzeET_P import analyzeET_P
from analyses.AnalyzeP_PET import analyzeP_PET
from analyses.AnalyzeP_ET import analyzeP_ET

# magnitude analyses
from analyses.AnalyzeMASD import analyzeMASD
from analyses.AnalyzeMASP import analyzeMASP
from analyses.AnalyzeMAT import analyzeMAT
from analyses.AnalyzeMASET import analyzeMASET
from analyses.AnalyzeMASPET import analyzeMASPET

# timing analyses
from analyses.AnalyzeDOMF import analyzeDOMF
from analyses.AnalyzeDOPF import analyzeDOPF

from analyses.AnalyzeDOMP import analyzeDOMP
from analyses.AnalyzeDOPP import analyzeDOPP

from analyses.AnalyzeDOMT import analyzeDOMT
from analyses.AnalyzeDOPT import analyzeDOPT

from analyses.AnalyzeDOMET import analyzeDOMET
from analyses.AnalyzeDOPET import analyzeDOPET

from analyses.AnalyzeDOMPET import analyzeDOMPET
from analyses.AnalyzeDOPPET import analyzeDOPPET

# spectral analyses
from analyses.AnalyzePOMMF import analyzePOMMF
from analyses.AnalyzePOMMP import analyzePOMMP
from analyses.AnalyzePOMMT import analyzePOMMT
from analyses.AnalyzePOMMET import analyzePOMMET
from analyses.AnalyzePOMMPET import analyzePOMMPET

# later stage analyses ********************************
from analyses.ImputeMetadata import imputeMetadata
from analyses.ImputeMetadataLog import imputeMetadataLog
from analyses.CombineResults import combineResults
from analyses.ComputeM import computeM
from analyses.PCAMetadata import pcaMetadata
from analyses.MakeTable import makeTable

# analzye correlations between variables *************
from analyses.AnalyzeCorrelations import analyzeCorrelations
from analyses.AnalyzeCorrelationsLinear import analyzeCorrelationsLinear
from analyses.AnalyzeCorrelationsNonlinear import analyzeCorrelationsNonlinear
from analyses.AnalyzeCorrelationsFigure import analyzeCorrelationsFigure


from analyses.AnalyzeCorrelationsPCA import analyzeCorrelationsPCA
from analyses.AnalyzeCorrelationsLinearPCA import analyzeCorrelationsLinearPCA
from analyses.AnalyzeCorrelationsNonLinearPCA import analyzeCorrelationsNonlinearPCA
from analyses.AnalyzeCorrelationsFigurePCA import analyzeCorrelationsFigurePCA


from analyses.MapAll import mapAll
from analyses.ExploratoryPlots import exploratoryPlots
from analyses.PlotFuh import plotFuh
from analyses.AnalyzeSpearmanCorrelations import analyzeSpearmanCorrelations

from analyses.ImputeChanges import imputeChanges

from analyses.plotDistributions import plotDistributions
from analyses.CalculateSensitivities import calculateSensitivities
from analyses.CalculateSensitivities2 import calculateSensitivities2
from analyses.CalculateBiomeVulnerabilities import calculateBiomeVulnerabilities
def main():
    runExploratoryAnalyses = True

    # adjust timeseries for ease of computation ***************************
    #combineTimeseries()
    #addLocalWaterYear()

    # calculate budget deficits
    #calculateBudgetDeficits()

    # ratio analyses ******************************************************
    #analyzeD_P() # discharge / precip (runoff ratio)
    #analyzePET_ET() #
    #analyzePET_P() # Budyko x-axis (aridity index)
    #analyzeET_P() # Budyko y-axis
    #analyzeP_PET() # Humidity Index
    #analyzeP_ET() # Humidity Index

    # magnitude of timeseries (mean annual) *******************************
    #analyzeMASD() # specific discharge
    #analyzeMASP() # specific precipitation
    #analyzeMAT() # temperature
    #analyzeMASET() # specific evapotranspiration
    #analyzeMASPET() # specific potential evapotranspiration

    # timing of timeseries (day of) ***************************************

    # mean
    #analyzeDOMF() # flow
    #analyzeDOMP() # precip
    #analyzeDOMT() # temperature
    #analyzeDOMET() # et
    #analyzeDOMPET() # pet

    # peak
    #analyzeDOPF() # flow
    #analyzeDOPP() # precip
    #analyzeDOPT() # temperature
    #analyzeDOPET() # et
    #analyzeDOPPET() # pet

    # spectral properties of timeseries (period of) ***********************
    #analyzePOMMF() # flow
    #analyzePOMMP() # precip
    #analyzePOMMT() # temperature
    #analyzePOMMET() # et
    #analyzePOMMPET() # pet

    # combine the data with the metadata
    #combineResults()
    #computeM()
    #imputeMetadata()
    #if runExploratoryAnalyses:
    #    pcaMetadata()

    # Identify important variables
    #analyzeCorrelations() # visual representation connections between variables
    #analyzeCorrelationsLinear() # linear regression analysis
    #analyzeCorrelationsNonlinear() # ml regression analysis
    analyzeCorrelationsFigure() # combine together into figures
    
    #if runExploratoryAnalyses:
        # Identify important PCA variables
    #    analyzeCorrelationsPCA() #isual representation connections between variables
    #    analyzeCorrelationsLinearPCA() # linear regression analysis
    #    analyzeCorrelationsNonlinearPCA() # ml regression analysis
    #    analyzeCorrelationsFigurePCA() # combine together into figures

    # make table of highest individual correlations for each factor
    #analyzeSpearmanCorrelations()
    #makeTable(numToShow=5)

    # global figures
    #imputeChanges() # impute missing predicted values
    #mapAll()
    #mapAll(randomForest=True)

    # plot changes
    #plotDistributions()
    #plotFuh()
    #if runExploratoryAnalyses:
    #    exploratoryPlots(colorVar="cls3", lowerBound=0, upperBound=1) # other examples include `colorVar="m"`, lowerbound=0, upperBound=10
    #calculateSensitivities(numIterations=1e1)
    #calculateSensitivities2()
    #calculateBiomeVulnerabilities()
    # FIXME: add % changes in sensitivity
    # FIXME: make a world map of potential flow

    print("all analyses complete")


if __name__ == "__main__":
    main()




