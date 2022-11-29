import statistics
import matplotlib.animation as animation
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from adferdir import Foll, Pendulum
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

def spurning3(plot=False):
    follin = Foll()
    p = Pendulum()
    y = follin.euler(p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)

    hnit = []
    for theta in y:
        hnit.append(follin.hornTohnit(theta))
    hnit = np.array(hnit)
    if plot:
        follin.create_animation2d(hnit)

def spurning4(plot=False):
    follin = Foll()
    p = Pendulum()
    y = follin.euler(p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=20)

    hnit = []
    for theta in y:
        hnit.append(follin.hornTohnit(theta))
    hnit = np.array(hnit)
    if plot:
        follin.create_animation2d(hnit)

def spurning5(plot=False):
    follin = Foll()
    p = Pendulum()
    y = follin.RKmethod(f=p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=10)
    hnit = []
    for theta in y:
        hnit.append(follin.hornTohnit(theta))
    hnit = np.array(hnit)

    if plot:
        follin.create_animation2d(hnit)

def spurning6():
    '''

    er á ipad pdf skjalinu hér á git.

    '''
    pass

def spurning7(plot=False):


    follin = Foll()
    p = Pendulum()
    p.double_pendulum1(1,2,3,4)
    y1, y2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=np.pi/3, horn2=np.pi/6, hornhradi1=0, hornhradi2=0, fjoldiskrefa=1000, lengd=20)
    hnit = []
    for theta in y1:
        hnit.append(follin.hornTohnit(theta))
    hnit = np.array(hnit)

    if plot:
        plt.plot(y1)
        plt.plot(y2)
        plt.show()
        #follin.create_animation2d(hnit)


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
    spurning1()
    spurning2()
    spurning3()
    spurning4()
    spurning5(plot=False)
    spurning7(plot=True)

