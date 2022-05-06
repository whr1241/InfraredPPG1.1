import matplotlib.pyplot as plt
import pywt
import numpy as np
from PyEMD import EEMD, EMD, Visualisation


s = np.random.random(100)
emd = EMD()
IMFs = emd.emd(s)

