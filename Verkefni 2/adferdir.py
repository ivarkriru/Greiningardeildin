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
from numpy import sin, cos

#f = lambda t, yinner: np.power(t, 2) +2
#f2 = lambda t, yinner: np.power(t, 2) +2
#f = lambda t, yinner: t * yinner - np.power(t, 3)

class Foll:
    def __init__(self,):
        self.c = 299792.458
        self.g = 9.81
        self.L = 2

    def euler(self,  f, horn, hornhradi, fjoldiskrefa, lengd):
        skreflengd = lengd / fjoldiskrefa
        skref = 0
        hornaxis = []
        hornhradiaxis = []

        demparastuðull = 0.00


        hornaxis.append(horn)
        hornhradiaxis.append(hornhradi+0.0000000001)

        for i in range(0,fjoldiskrefa):
            skref = skref + skreflengd
            hornaxis.append(hornaxis[i] + skreflengd*hornhradiaxis[i])
            dempun = -1*demparastuðull*(hornhradiaxis[i]/(abs(hornhradiaxis[i])))
            hornhradiaxis.append(hornhradiaxis[i] + skreflengd*f(hornaxis[i]) + dempun)

        return hornaxis

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
            s1 = skreflengd * f(hornaxis[i])
            s2 = skreflengd * f(hornaxis[i] + skreflengd*(s1/2))
            s3 = skreflengd * f(hornaxis[i] + skreflengd*(s2/2))
            s4 = skreflengd * f(hornaxis[i] + skreflengd * s3)
            w = hornhradiaxis[i] + (s1+s2*2+s3*2+s4)/6
            hornhradiaxis.append(w)
            hornaxis.append(hornaxis[i] + skreflengd*w)

        return hornaxis

    def RKmethod2(self, f1, f2, horn1, horn2, hornhradi1, hornhradi2, fjoldiskrefa, lengd):
        skreflengd = lengd / fjoldiskrefa  # h = skreflengd
        skref = 0  # skref = t
        horn1axis = []
        horn1hradiaxis = []
        horn2axis = []
        horn2hradiaxis = []

        dempari = 0
        horn1axis.append(horn1)
        horn1hradiaxis.append(hornhradi1)
        horn2axis.append(horn2)
        horn2hradiaxis.append(hornhradi2)

        for i in range(0, fjoldiskrefa):
            skref = skref + skreflengd
            s1 = skreflengd * f1(horn1axis[i], horn2axis[i], horn1hradiaxis[i], horn2hradiaxis[i])
            s2 = skreflengd * f1(horn1axis[i] + skreflengd*(s1/2), horn2axis[i] + skreflengd*(s1/2), horn1hradiaxis[i], horn2hradiaxis[i])
            s3 = skreflengd * f1(horn1axis[i] + skreflengd*(s2/2), horn2axis[i] + skreflengd*(s2/2), horn1hradiaxis[i], horn2hradiaxis[i])
            s4 = skreflengd * f1(horn1axis[i] + skreflengd*s3, horn2axis[i] + skreflengd*s3, horn1hradiaxis[i], horn2hradiaxis[i])
            w = horn1hradiaxis[i] + (s1+s2*2+s3*2+s4)/6
            horn1hradiaxis.append(w)
            horn1axis.append(horn1axis[i] + skreflengd*w)

            s1 = skreflengd * f2(horn1axis[i], horn2axis[i], horn1hradiaxis[i], horn2hradiaxis[i])
            s2 = skreflengd * f2(horn1axis[i] + skreflengd*(s1/2), horn2axis[i] + skreflengd*(s1/2), horn1hradiaxis[i], horn2hradiaxis[i])
            s3 = skreflengd * f2(horn1axis[i] + skreflengd*(s2/2), horn2axis[i] + skreflengd*(s2/2), horn1hradiaxis[i], horn2hradiaxis[i])
            s4 = skreflengd * f2(horn1axis[i] + skreflengd*s3, horn2axis[i] + skreflengd*s3, horn1hradiaxis[i], horn2hradiaxis[i])
            w = horn2hradiaxis[i] + (s1+s2*2+s3*2+s4)/6
            horn2hradiaxis.append(w)
            horn2axis.append(horn2axis[i] + skreflengd*w)

        return horn1axis, horn2axis

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

class Pendulum:
    def __init__(self,L_1=2,m_1=1,L_2=2,m_2=1):
        self.L_1 = L_1
        self.m_1 = m_1
        self.L_2 = L_2
        self.m_2 = m_2
        self.g = 9.81

    def pendulum(self, theta):
        fasti = -1*self.g/(self.L_1)
        return np.sin(theta) * fasti

    def double_pendulum1(self, theta1, theta2, omega1, omega2):
        l1 = self.L_1
        l2 = self.L_2
        m1 = self.m_1
        m2 = self.m_1
        g = self.g
        d = theta2 - theta1
        theta1_2prime = m2 * l1 * omega1**2 * sin(d) * cos(d) + m2 * g * sin(theta2)*cos(d) + m2 * l2 * omega2**2 * sin(d) - (m1 + m2)*g*sin(theta1
                            ) / (m1 + m2) * l1 - m2 * l1 * cos(d)**2

        return theta1_2prime

    def double_pendulum2(self, theta1, theta2, omega1, omega2):
        l1 = self.L_1
        l2 = self.L_2
        m1 = self.m_1
        m2 = self.m_1
        g = self.g
        d = theta2 - theta1



        theta2_2prime = -m2 * l1 * omega2**2 * sin(d) * cos(d) + (m1 + m2) * (g * sin(theta1)*cos(d)) + l1 * omega1**2 * sin(d) - g*sin(theta2
                        ) / (m1 + m2) * l2 - m2 * l2 * cos(d)**2

        return theta2_2prime

