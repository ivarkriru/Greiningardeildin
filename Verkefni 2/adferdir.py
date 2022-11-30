import random

import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import io
import os


g = 9.81

class Foll:
    def __init__(self, ):
        self.g = g
        self.L = 2

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
            s1 = skreflengd * f(hornaxis[i])
            s2 = skreflengd * f(hornaxis[i] + skreflengd * (s1 / 2))
            s3 = skreflengd * f(hornaxis[i] + skreflengd * (s2 / 2))
            s4 = skreflengd * f(hornaxis[i] + skreflengd * s3)
            w = hornhradiaxis[i] + (s1 + s2 * 2 + s3 * 2 + s4) / 6
            hornhradiaxis.append(w)
            hornaxis.append(hornaxis[i] + skreflengd * w)

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
            s2 = skreflengd * f1(horn1axis[i] + skreflengd * (s1 / 2), horn2axis[i] + skreflengd * (s1 / 2),
                                 horn1hradiaxis[i], horn2hradiaxis[i])
            s3 = skreflengd * f1(horn1axis[i] + skreflengd * (s2 / 2), horn2axis[i] + skreflengd * (s2 / 2),
                                 horn1hradiaxis[i], horn2hradiaxis[i])
            s4 = skreflengd * f1(horn1axis[i] + skreflengd * s3, horn2axis[i] + skreflengd * s3, horn1hradiaxis[i],
                                 horn2hradiaxis[i])
            w = horn1hradiaxis[i] + (s1 + s2 * 2 + s3 * 2 + s4) / 6
            horn1hradiaxis.append(w)
            horn1axis.append(horn1axis[i] + skreflengd * w)

            s1 = skreflengd * f2(horn1axis[i], horn2axis[i], horn1hradiaxis[i], horn2hradiaxis[i])
            s2 = skreflengd * f2(horn1axis[i] + skreflengd * (s1 / 2), horn2axis[i] + skreflengd * (s1 / 2),
                                 horn1hradiaxis[i], horn2hradiaxis[i])
            s3 = skreflengd * f2(horn1axis[i] + skreflengd * (s2 / 2), horn2axis[i] + skreflengd * (s2 / 2),
                                 horn1hradiaxis[i], horn2hradiaxis[i])
            s4 = skreflengd * f2(horn1axis[i] + skreflengd * s3, horn2axis[i] + skreflengd * s3, horn1hradiaxis[i],
                                 horn2hradiaxis[i])
            w = horn2hradiaxis[i] + (s1 + s2 * 2 + s3 * 2 + s4) / 6
            horn2hradiaxis.append(w)
            horn2axis.append(horn2axis[i] + skreflengd * w)

        return horn1axis, horn2axis


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

        #k1 = m2 * l1 * math.pow(omega1, 2) * np.sin(d) * np.cos(d)
        #k2 = m2 * g * np.sin(theta2) * np.cos(d)
        #k3 = m2 * l2 * math.pow(omega2, 2) * np.sin(d)
        #k4 = (m1 + m2) * g * np.sin(theta1)
        #k5 = (m1 + m2) * l1 - m2 * l1 * math.pow(np.cos(d), 2)

        k1 = m2*l2*math.pow(omega2,2)*np.sin(d)
        k2 =-((m1+m2)*g*np.sin(theta1))
        k3 = m2*l1*math.pow(omega1,2)*np.sin(d)*np.cos(d)
        k4 = -m2*g*np.sin(theta2)*np.cos(d)
        k5 = ((m1+m2)*l1 - m2*l1*np.cos(d)*np.cos(d))

        if l1 == 0 or k5 == 0 or (k1 + k2 + k3 - k4) == 0:
            return 0

        theta1_2prime = (k1 + k2 + k3 - k4) / k5
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

        #k1 = m2 * l2 * math.pow(omega2, 2) * np.sin(d) * np.cos(d)
        #k2 = (m1 + m2) * ( g * np.sin(theta1) * np.cos(d))
        #k3 = l1 * math.pow(omega1, 2) * np.sin(d)
        #k4 = g * np.sin(theta2)
        #k5 = (m1 + m2) * l2 - m2 * l2 * math.pow(np.cos(d), 2)

        k1 = -m2*l2*math.pow(omega2,2)*np.sin(d)*np.cos(d)
        k2 = -((m1+m2)*g*np.sin(theta1))*np.cos(d)
        k3 = -(m1+m2)*l1*math.pow(omega1,2)*np.sin(d)
        k4 = -(m1+m2)*g*np.sin(theta2)
        k5 = (m2*l2*np.cos(d)*np.cos(d) - (m1+m2)*l2)

        if l2 == 0 or k5 == 0 or (k1 + k2 + k3 - k4) == 0:
            return 0

        theta2_2prime = (-k1 + k2 - k3 - k4) / k5
        return theta2_2prime

    def hnitforanimationusingEuler(self, fall, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20):
        follin = Foll()
        y = Foll().euler(fall, horn, hornhradi, fjoldiskrefa, lengd)
        hnit = []
        for theta in y:
            hnit.append(self.hornTohnit(theta))
        hnit = np.array(hnit)
        return hnit, y

    def hnitforanimationusingRK(self, fall, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20):
        y = Foll().RKmethod(f=fall, horn=horn, hornhradi=hornhradi, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
        hnit = []
        for theta in y:
            hnit.append(self.hornTohnit(theta))
        hnit = np.array(hnit)

        return hnit,y

    def hornTohnit(self, th, L_1=None):
        if L_1 is None:
            L_1 = self.L_1
        return L_1 * np.sin(th), -L_1 * np.cos(th)

    def hornTohnitjunior(self, th, th2, L_1=None, L_2=None):
        if L_1 is None:
            L_1=self.L_1
        if L_2 is None:
            L_2=self.L_2
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
                plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))
                x = data1[index, 0]
                y = data1[index, 1]
                x2 = data2[index, 0]
                y2 = data2[index, 1]


                plt.plot([-staerdramma * 2, -staerdramma * 2], [0, 0], lw=1, c="black")

                plt.scatter(x, y, lw=20, c="orange")
                plt.plot([0, x], [0, y], lw=5, c="blue")
                plt.scatter(x2, y2, lw=20, c="red")
                plt.plot([0, x2], [0, y2], lw=5, c="green")
                plt.xticks([])
                plt.yticks([])


                plt.pause(0.001)

        plt.show()

    def create_animation2d(self, data1, data2=None, fjoldipendula=1, title=None, savegif=False):

        # initializing a figure in
        # which the graph will be plotted
        # marking the x-axis and y-axis
        staerdramma = self.L_2 + self.L_1 + 3
        bufs = []

        if fjoldipendula == 1:
            for index in range(data1.shape[0]):

                plt.clf()
                plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))
                plt.title(label=title)
                x = data1[index, 0]
                y = data1[index, 1]

                plt.plot([-staerdramma, staerdramma], [0, 0], lw=4, c="black")

                plt.scatter(x, y, lw=20, c="orange")
                plt.plot([0, x], [0, y], lw=5, c="blue")

                plt.pause(0.001)
        elif fjoldipendula == 2:
            for index in range(0, data1.shape[0], 2):
                plt.clf()
                plt.title(label=title)
                plt.axes(xlim=(-staerdramma, staerdramma), ylim=(-staerdramma, staerdramma))

                x1 = data1[index, 0]
                y1 = data1[index, 1]

                x2 = data2[index, 0]
                y2 = data2[index, 1]


                x1plot = data1[0:index, 0]
                y1plot = data1[0:index, 1]

                x2plot = data2[0:index, 0]
                y2plot = data2[0:index, 1]

                plt.plot([-staerdramma * 2, staerdramma * 2], [0, 0], lw=3, c="black")

                plt.plot(x1plot, y1plot, c="red")
                plt.plot(x2plot, y2plot, c="red")

                plt.plot([0, x1], [0, y1], lw=5, c="blue")
                plt.plot([x1, x2], [y1, y2], lw=5, c="blue")

                plt.scatter(x1, y1, lw=self.m_1 * 5, c="orange")
                plt.scatter(x2, y2, lw=self.m_2 * 5, c="orange")
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
