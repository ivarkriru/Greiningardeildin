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

f = lambda t, yinner: np.power(t, 2) +2

class foll:
    def __init__(self):
        self.c = 299792.458

    def euler(y0,n,T):
        h = T/n
        t = 0
        xaxis = []
        yaxis = []
        xaxis.append(t)
        yaxis.append(y0)
        y=y0
        for i in range(1, n):
            t = t + h
            xaxis.append(t)
            yaxis.append(yaxis[i-1] + f(t,yaxis[i-1]) + h)

        x = range(y0,T)
        y=f(x,x)
        plt.plot(x,y)
        plt.plot(xaxis,yaxis)
        np.linspace(0,5)
        plt.show()
