import numpy as np
import matplotlib.pyplot as plt
import statistics
import matplotlib.animation as animation
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
import os
from matplotlib import animation
from scipy import stats as stats

#f = lambda t, yinner: np.power(t, 2) +2
#f2 = lambda t, yinner: np.power(t, 2) +2
#f = lambda t, yinner: t * yinner - np.power(t, 3)

class foll:
    def __init__(self):
        self.c = 299792.458
        self.g = 9.81
        self.L = 2

    def euler(self, horn, hornhradi, fjoldiskrefa, lengd):
        skreflengd = lengd / fjoldiskrefa
        skref = 0
        hornaxis = []
        hornhradiaxis = []

        dempari = 0

        fasti = -1*self.g/(self.L)

        hornaxis.append(horn)
        hornhradiaxis.append(hornhradi)

        for i in range(0,fjoldiskrefa):
            skref = skref + skreflengd
            hornaxis.append(hornaxis[i] + skreflengd*hornhradiaxis[i])
            hornhradiaxis.append(hornhradiaxis[i] + skreflengd*(np.sin(hornaxis[i]))*fasti - dempari)

        return hornaxis
