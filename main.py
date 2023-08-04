from analyses.AnalyzeRunoffRatio import analyzeRunoffRatio
from analyses.AnalyzeMASD import analyzeMASD
from analyses.AnalyzeDOMF import analyzeDOMF
from analyses.AnalyzePeakFlows import analyzePeakFlows
from analyses.AnalyzeFourier import analyzeFourier

def main():
    analyzeRunoffRatio()
    analyzeMASD()
    analyzeDOMF()
    analyzePeakFlows()
    analyzeFourier()

if __name__ == "__main__":
    main()




