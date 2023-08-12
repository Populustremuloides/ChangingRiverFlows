from scipy.stats import theilslopes
import numpy as np
from data.metadata import *

# utility functions start with u_*

u_regressionFunction = theilslopes

def u_getCatchmentName(fileName):
    cat = fileName.split(".")[0]
    cat = cat.split("_")[-1]
    return cat

