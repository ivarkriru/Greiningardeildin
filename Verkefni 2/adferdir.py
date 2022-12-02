import random

import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io
import os
import subprocess

from matplotlib import animation

g = 9.81

class Foll:
    def __init__(self, ):
        self.g = g

    def euler(self, f, horn, hornhradi, fjoldiskrefa, lengd):
        skreflengd = lengd / fjoldiskrefa
        skref = 0
        hornaxis = []
        hornhradiaxis = []

        demparastuðull = 0.00

        hornaxis.append(horn)
        hornhradiaxis.append(hornhradi + 0.0000000001)

        for i in range(0, fjoldiskrefa):
            skref = skref + skreflengd
            hornaxis.append(hornaxis[i] + skreflengd * hornhradiaxis[i])
            dempun = -1 * demparastuðull * (hornhradiaxis[i] / (abs(hornhradiaxis[i])))
            hornhradiaxis.append(hornhradiaxis[i] + skreflengd * f(hornaxis[i]) + dempun)

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
            s1 =  f(hornaxis[i])
            s2 =  f(hornaxis[i] + skreflengd * (s1 / 2))
            s3 =  f(hornaxis[i] + skreflengd * (s2 / 2))
            s4 =  f(hornaxis[i] + skreflengd * s3)

            w = hornhradiaxis[i] + (s1 + s2 * 2 + s3 * 2 + s4) / 6 * skreflengd
            hornhradiaxis.append(w)
            hornaxis.append(hornaxis[i] + skreflengd * w)

        return hornaxis

    def RKmethod2(self, f1, f2, horn1, horn2, hornhradi1, hornhradi2, fjoldiskrefa, lengd):
        skreflengd = lengd / fjoldiskrefa  # h = skreflengd
        skref = 0  # skref = t
        axis = np.zeros((fjoldiskrefa+1, 4))
        axis[0] = np.array([[horn1, horn2, hornhradi1, hornhradi2]])

        def f(y):
            th1, th2, thp1, thp2 = y

            return np.array([thp1, thp2, f1(*y), f2(*y)])

        for i in range(0, fjoldiskrefa):
            skref = skref + skreflengd

            s1 = f(axis[i])
            s2 = f((axis[i] + skreflengd*s1/2))
            s3 = f((axis[i] + skreflengd*s2/2))
            s4 = f((axis[i] + skreflengd*s3))

            axis[i+1] = axis[i] + (s1 + s2 * 2 + s3 * 2 + s4) / 6 * skreflengd


        return axis


class Pendulum:
    def __init__(self, L_1=2, m_1=1, L_2=2, m_2=1):
        self.L_1 = L_1
        self.m_1 = m_1
        self.L_2 = L_2
        self.m_2 = m_2
        self.g = g

    def pendulum(self, theta):
        fasti = -1 * self.g / (self.L_1)
        return np.sin(theta) * fasti

    def double_pendulum1(self, theta1, theta2, omega1, omega2):
        l1 = self.L_1
        l2 = self.L_2
        m1 = self.m_1
        m2 = self.m_2
        g = self.g
        d = theta2 - theta1

        if omega1 > 2e+50:
            omega1 = 2e+50
        if omega2 > 2e+50:
            omega2 = 2e+50

        if omega1 < -2e+50:
            omega1 = -2e+50
        if omega2 < -2e+50:
            omega2 = -2e+50

        cosd = math.cos(d)
        sind = math.sin(d)
        k1 = m2*l1*math.pow(omega1,2)*sind*cosd
        k2 = m2*g*math.sin(theta2)*cosd
        k3 = m2*l2*math.pow(omega2,2)*sind
        k4 = -((m1+m2)*g*math.sin(theta1))
        k5 = ((m1+m2)*l1 - m2*l1*cosd*cosd)

        if l1 == 0 or k5 == 0 or (k1 + k2 + k3 + k4) == 0:
            return 0

        theta1_2prime = (k1 + k2 + k3 + k4) / k5
        return theta1_2prime

    def double_pendulum2(self, theta1, theta2, omega1, omega2):
        l1 = self.L_1
        l2 = self.L_2
        m1 = self.m_1
        m2 = self.m_2
        g = self.g
        d = theta2 - theta1

        if omega1 > 2e+50:
            omega1 = 2e+50
        if omega2 > 2e+50:
            omega2 = 2e+50

        if omega1 < -2e+50:
            omega1 = -2e+50
        if omega2 < -2e+50:
            omega2 = -2e+50
        cosd = math.cos(d)
        sind = math.sin(d)
        k1 = -m2*l2*math.pow(omega2,2)*sind*cosd
        k2 = ((m1+m2)*g*math.sin(theta1))*cosd
        k3 = -(m1+m2)*l1*math.pow(omega1,2)*sind
        k4 = -(m1+m2)*g*math.sin(theta2)
        k5 = (m1+m2)*l2 - m2*l2 * cosd

        if l2 == 0 or k5 == 0 or (k1 + k2 + k3 + k4) == 0:
            return 0

        theta2_2prime = (k1 + k2 + k3 + k4) / k5
        return theta2_2prime

    def hnitforanimationusingEuler(self, fall, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20):
        follin = Foll()
        y = Foll().euler(fall, horn, hornhradi, fjoldiskrefa, lengd)
        hnit = []
        for theta in y:
            hnit.append(self.hornTohnit(theta))
        hnit = np.array(hnit)
        #y = np.array(y) * (180 / np.pi)
        return hnit, y

    def hnitforanimationusingRK(self, fall, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20):
        y = Foll().RKmethod(f=fall, horn=horn, hornhradi=hornhradi, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
        hnit = []
        for theta in y:
            hnit.append(self.hornTohnit(theta))
        hnit = np.array(hnit)
        y = np.array(y) * (180 / np.pi)

        return hnit,y

    def hnitforanimationusingRK2(self, L_1=2, m_1=1, L_2=2, m_2=1, horn1=np.pi * 3 / 4,
                                  horn2=np.pi * 6 / 4,
                                  hornhradi1=1, hornhradi2=0, fjoldiskrefa=100, lengd=100):
        follin = Foll()
        p = Pendulum(L_1=L_1, m_1=m_1, L_2=L_2, m_2=m_2)
        arr = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1= horn1,
                                  horn2= horn2,
                                  hornhradi1= hornhradi1, hornhradi2= hornhradi2, fjoldiskrefa=fjoldiskrefa * 30, lengd=lengd)

        y1 = arr[:,0]
        y2 = arr[:,1]

        hnitsenior = []
        hnitjunior = []

        for theta in y1:
            hnitsenior.append(p.hornTohnit(theta))
        for index, theta in enumerate(y2):
            hnitjunior.append(p.hornTohnitjunior(y1[index], theta))

        hnitsenior = np.array(hnitsenior)
        hnitjunior = np.array(hnitjunior)
        y1 = np.array(y1) * (180 / np.pi)
        y2 = np.array(y2) * (180 / np.pi)
        return hnitsenior,hnitjunior,y1,y2

    def hornTohnit(self, th):
        L_1 = self.L_1
        return L_1 * np.sin(th), -L_1 * np.cos(th)

    def hornTohnitjunior(self, th, th2):
        L_1 = self.L_1
        L_2 = self.L_2
        return L_1 * np.sin(th) + L_2 * np.sin(th2), -L_1 * np.cos(th) - L_2 * np.cos(th2)

    def create_animation2dfyrir4(self, data1, data2=None, fjoldipendula=1, title=None):
        # initializing a figure in
        # which the graph will be plotted
        # marking the x-axis and y-axis
        staerdramma = self.L_2 + self.L_1 + 3
        plt.axis('equal')

        plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))
        if fjoldipendula == 1:
            for index in range(data1.shape[0]):
                plt.clf()
                plt.title(label=title)

                x = data1[index, 0]
                y = data1[index, 1]
                x2 = data2[index, 0]
                y2 = data2[index, 1]
                plt.xticks([])
                plt.yticks([])
                plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))
                plt.xlabel('Staðsetning á x-ás í radíönum')
                plt.ylabel('Staðsetning á y-ás í radíönum')
                plt.plot([-staerdramma, staerdramma], [0, 0], lw=4, c="black")
                plt.scatter(x, y, lw=20, c="orange")
                plt.plot([0, x], [0, y], lw=5, c="blue",alpha=1)
                plt.scatter(x2, y2, lw=20, c="red")
                plt.plot([0, x2], [0, y2], lw=5, c="green")
                plt.pause(0.001)
        plt.show()

    def create_animation2d(self, data1, data2=None, fjoldipendula=1, title=None, savegif=False,trace=True):

        # initializing a figure in
        # which the graph will be plotted
        # marking the x-axis and y-axis
        staerdramma = self.L_2 + self.L_1 + 3

        bufs = []

        if fjoldipendula == 1:
            for index in range(data1.shape[0]):
                plt.clf()
                plt.title(label=title)
                x = data1[index, 0]
                y = data1[index, 1]
                plt.xticks([])
                plt.yticks([])
                plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))
                plt.xlabel('Staðsetning á x-ás í radíönum')
                plt.ylabel('Staðsetning á y-ás í radíönum')

                plt.plot([-staerdramma, staerdramma], [0, 0], lw=4, c="black")

                if trace:
                    plt.scatter(x, y, lw=20, c="orange")
                    plt.plot([0, x], [0, y], lw=5, c="blue")

                plt.pause(0.001)

        elif fjoldipendula == 2:
            for index in range(0, data1.shape[0], 50):
                plt.clf()
                plt.title(label=title)


                x1 = data1[index, 0]
                y1 = data1[index, 1]
                plt.xticks([])
                plt.yticks([])
                x2 = data2[index, 0]
                y2 = data2[index, 1]

                plt.xticks([])
                plt.yticks([])
                plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))

                plt.xlabel('Staðsetning á x-ás í radíönum')
                plt.ylabel('Staðsetning á y-ás í radíönum')
                x1plot = data1[0:index, 0]
                y1plot = data1[0:index, 1]

                x2plot = data2[0:index, 0]
                y2plot = data2[0:index, 1]

                plt.plot([-staerdramma * 2, staerdramma * 2], [0, 0], lw=3, c="black")
                if trace:
                    plt.plot(x1plot, y1plot, c="yellow")
                    plt.plot(x2plot, y2plot, c="cyan")

                plt.plot([0, x1], [0, y1], lw=5, c="blue")
                plt.plot([x1, x2], [y1, y2], lw=5, c="green")

                plt.scatter(x1, y1, lw=self.m_1 * 5, c="orange")
                plt.scatter(x2, y2, lw=self.m_2 * 5, c="red")
                if savegif:
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    bufs.append(Image.open(buf))
                print(f"{index / data1.shape[0]* 100:.00f}%", end="\n", flush=True)
                plt.pause(0.001)
        print("here")
        if savegif:
            filename = "animation_" + str(random.randint(0,10000)) + "_gif.gif"
            f = os.path.join(os.getcwd(), filename)
            bufs[0].save(f, save_all = True, append_images=bufs[1:], optimize=False, duration = 10)
        plt.show()
    def create_animation2ex2(self, data1, data2, data3, data4, fjoldipendula=1, title=None, savegif=False,offset = 5):

        # initializing a figure in
        # which the graph will be plotted
        # marking the x-axis and y-axis
        staerdramma = self.L_2*2 + self.L_1*2 + 3

        for index in range(0, data1.shape[0], 100):
            plt.clf()
            plt.title(label=title)


            x1 = data1[index, 0] + offset
            y1 = data1[index, 1]

            x2 = data2[index, 0] + offset
            y2 = data2[index, 1]

            x3 = data3[index, 0] - offset
            y3 = data3[index, 1]

            x4 = data4[index, 0] - offset
            y4 = data4[index, 1]
            plt.xticks([])
            plt.yticks([])
            plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))

            plt.xlabel('Staðsetning á x-ás í radíönum')
            plt.ylabel('Staðsetning á y-ás í radíönum')
            x1plot = data1[0:index, 0]
            x1plot = x1plot + offset
            y1plot = data1[0:index, 1]

            x2plot = data2[0:index, 0]
            x2plot = x2plot + offset
            y2plot = data2[0:index, 1]

            x3plot = data3[0:index, 0]
            x3plot = x3plot - offset
            y3plot = data3[0:index, 1]

            x4plot = data4[0:index, 0]
            x4plot = x4plot - offset
            y4plot = data4[0:index, 1]

            plt.plot([-staerdramma * 2, staerdramma * 2], [0, 0], lw=3, c="black")

            plt.plot(x1plot, y1plot)
            plt.plot(x2plot, y2plot)
            plt.plot(x3plot, y3plot)
            plt.plot(x4plot, y4plot)

            plt.plot([offset, x1], [0, y1], lw=5, c="blue")
            plt.plot([x1, x2], [y1, y2], lw=5, c="blue")
            plt.plot([-offset,x3], [0, y3], lw=5, c="blue")
            plt.plot([x3, x4], [y3, y4], lw=5, c="blue")

            plt.scatter(x1, y1, lw=self.m_1 * 5*2, c="orange")
            plt.scatter(x2, y2, lw=self.m_2 * 5*2, c="orange")
            plt.scatter(x3, y3, lw=self.m_1 * 5*2, c="orange")
            plt.scatter(x4, y4, lw=self.m_2 * 5*2, c="orange")

            plt.pause(0.001)

            print(f"{index / data1.shape[0] * 100:.00f}%", end="\n", flush=True)
        plt.show()

