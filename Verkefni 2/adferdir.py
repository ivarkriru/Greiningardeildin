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
from matplotlib.animation import FuncAnimation

#f = lambda t, yinner: np.power(t, 2) +2
#f2 = lambda t, yinner: np.power(t, 2) +2
#f = lambda t, yinner: t * yinner - np.power(t, 3)

class Foll:
    def __init__(self):
        self.c = 299792.458
        self.g = 9.81
        self.L = 2

    def euler(self, horn, hornhradi, fjoldiskrefa, lengd):
        skreflengd = lengd / fjoldiskrefa
        skref = 0
        hornaxis = []
        hornhradiaxis = []

        demparastuðull = 0.00

        fasti = -1*self.g/(self.L)

        hornaxis.append(horn)
        hornhradiaxis.append(hornhradi+0.0000000001)

        for i in range(0,fjoldiskrefa):
            skref = skref + skreflengd
            hornaxis.append(hornaxis[i] + skreflengd*hornhradiaxis[i])
            dempun = -1*demparastuðull*(hornhradiaxis[i]/(abs(hornhradiaxis[i])))
            hornhradiaxis.append(hornhradiaxis[i] + skreflengd*(np.sin(hornaxis[i]))*fasti + dempun)

        return hornaxis
    def pendulum(self, theta):
        fasti = -1*self.g/(self.L)
        return np.sin(theta) * fasti

    def RKmethod(self, f, horn, hornhradi, fjoldiskrefa, lengd):
        skreflengd = lengd / fjoldiskrefa  # h = skreflengd
        skref = 0  # skref = t
        hornaxis = []
        hornhradiaxis = []

        dempari = 0
        hornaxis.append(horn)
        hornhradiaxis.append(hornhradi)

        for i in range(0, fjoldiskrefa):
            skref = skref + skreflengd

            s1 = f(hornaxis[i])
            s2 = f(hornaxis[i] + skreflengd*s1/2)
            s3 = f(hornaxis[i] + skreflengd*s2/2)
            s4 = f(hornaxis[i] + skreflengd * s3)

            hornaxis.append(hornaxis[i] + skreflengd*(s1/6+s2/3+s3/3+s4/6))
            hornhradiaxis.append(f(hornaxis[i]))
        return hornaxis

    def hornTohnit(self,th,th2=0):
        return self.L * np.sin(th),-self.L*np.cos(th)


    def create_animation2d(self,data1,data2=None,fjoldipendula=1):

        # initializing a figure in
        # which the graph will be plotted
        # marking the x-axis and y-axis
        plt.axis('equal')
        plt.axes(xlim=(-self.L * 1.5, self.L * 1.5), ylim=(-self.L * 1.5, self.L * 1.5))

        if fjoldipendula == 1:
            for index in range(data1.shape[0]):
                plt.clf()
                plt.axes(xlim=(-self.L*1.5, self.L*1.5),ylim=(-self.L*1.5, self.L*1.5))
                x = data1[index, 0]
                y = data1[index, 1]
                plt.scatter(x,y,lw=20,c="orange")
                plt.plot([0,x],[0,y],lw=5,c="blue")
                plt.plot([-10,10],[0,0],lw=1,c="black")

                plt.pause(0.001)

        elif fjoldipendula == 2:
            for index in range(data1.shape[0]):
                plt.clf()
                plt.axes(xlim=(-self.L * 1.5, self.L * 1.5), ylim=(-self.L * 1.5, self.L * 1.5))
                x = data1[index, 0]
                y = data1[index, 1]
                plt.scatter(x, y, lw=20, c="orange")
                plt.plot([0, x], [0, y], lw=5, c="blue")
                plt.plot([-10, 10], [0, 0], lw=1, c="black")

                plt.pause(0.001)

        plt.show()



