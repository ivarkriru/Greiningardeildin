import statistics
import matplotlib.animation as animation
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from adferdir import foll
import time
import os
from matplotlib import animation
from scipy import stats as stats


if __name__ == '__main__':
    #spurning1()
    #spurning2()
    #spurning3()

    follin = foll()

    y = follin.euler(horn=np.pi/2, hornhradi=0, fjoldiskrefa=500, lengd=20)
    x = range(0,len(y))
    plt.plot(x,y)
    plt.show()
