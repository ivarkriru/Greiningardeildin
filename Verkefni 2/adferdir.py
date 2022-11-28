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

f = lambda yinner: np.array(yinner, (-g/L * np.sin(yinner)))

class foll:
    def __init__(self):
        self.c = 299792.458
        self.g = 9.81
        self.L = 2

    def euler(self, y0, n, T):
        h = T / n
        t = 0
        xaxis = []
        yaxis = []
        xaxis.append(t)
        yaxis.append(y0)
        y = y0
        for i in range(1, n + 1):
            t = t + h
            xaxis.append(t)
            yaxis.append(yaxis[i - 1] + f(yaxis[i - 1]) * h)
            plt.plot(xaxis, yaxis)
            #    np.linspace(0,5)
        return xaxis


