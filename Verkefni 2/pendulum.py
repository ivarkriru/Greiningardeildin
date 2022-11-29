import statistics
import matplotlib.animation as animation
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from adferdir import Foll
import time
import os
from matplotlib import animation
from scipy import stats as stats

def spurning1():
    '''

    er á ipad pdf skjalinu hér á git.

    '''
    pass

def spurning2():
    '''

    euler fallið í adferd python skjalinu

    '''
    pass

def spurning3():
    follin = Foll()

    y = follin.euler(horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)

    hnit = []
    for theta in y:
        hnit.append(follin.hornTohnit(theta))
    hnit = np.array(hnit)

    follin.create_animation2d(hnit)

def spurning4():
    follin = Foll()

    y = follin.euler(horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=20)

    hnit = []
    for theta in y:
        hnit.append(follin.hornTohnit(theta))
    hnit = np.array(hnit)

    follin.create_animation2d(hnit)

def spurning5():
    follin = Foll()
    y = follin.RKmethod(f=follin.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=10)
    hnit = []
    for theta in y:
        hnit.append(follin.hornTohnit(theta))
    hnit = np.array(hnit)

    #print(y)
    #plt.plot(y)
    #plt.show()

    follin.create_animation2d(hnit)

def spurning6():
    pass

def spurning7():
    pass

def spurning8():
    pass

def spurning9():
    pass

def spurning10():
    pass

def spurning11():
    pass

def spurning12():
    pass

def frjals():
    pass

if __name__ == '__main__':
    #spurning1()
    #spurning2()
    #spurning3()
    # spurning4()
    spurning5()

