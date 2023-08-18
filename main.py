from analyses.AddLocalWaterYear import addLocalWaterYear

# ratio analyses
from analyses.AnalyzeD_P import analyzeD_P
from analyses.AnalyzePET_ET import analyzePET_ET
from analyses.AnalyzePET_P import analyzePET_P
from analyses.AnalyzeET_P import analyzeET_P
from analyses.AnalyzeP_PET import analyzeP_PET

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
from analyses.CombineResults import combineResults

def main():
    # adjust timeseries for ease of computation ***************************
    #addLocalWaterYear()

    # ratio analyses ******************************************************
    #analyzeD_P() # discharge / precip (runoff ratio)
    #analyzePET_ET() #
    #analyzePET_P() # Budyko x-axis (aridity index)
    #analyzeET_P() # Budyko y-axis
    #analyzeP_PET() # Humidity Index


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

    # linear regression analysis

    # relationships with important variables

    # distribution figures

    # global figures

if __name__ == "__main__":
    main()




