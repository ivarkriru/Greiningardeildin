import random

# import numpy as np
import math
#import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
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

    def RKmethod2(self, f1, f2, horn1, horn2, hornhradi1, hornhradi2, fjoldiskrefa, lengd, sp9=False):
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
            s1 = f1(horn1axis[i], horn2axis[i], horn1hradiaxis[i], horn2hradiaxis[i])
            s2 = f1(horn1axis[i] + skreflengd * (s1 / 2), horn2axis[i] + skreflengd * (s1 / 2),
                    horn1hradiaxis[i], horn2hradiaxis[i])
            s3 = f1(horn1axis[i] + skreflengd * (s2 / 2), horn2axis[i] + skreflengd * (s2 / 2),
                    horn1hradiaxis[i], horn2hradiaxis[i])
            s4 = f1(horn1axis[i] + skreflengd * s3, horn2axis[i] + skreflengd * s3, horn1hradiaxis[i],
                                 horn2hradiaxis[i])
            w = horn1hradiaxis[i] + (s1 + s2 * 2 + s3 * 2 + s4) / 6 * skreflengd
            horn1hradiaxis.append(w)
            horn1axis.append((horn1axis[i] + skreflengd * w) % (math.pi*2))

            s1 = f2(horn1axis[i], horn2axis[i], horn1hradiaxis[i], horn2hradiaxis[i])
            s2 = f2(horn1axis[i] + skreflengd * (s1 / 2), horn2axis[i] + skreflengd * (s1 / 2),
                    horn1hradiaxis[i], horn2hradiaxis[i])
            s3 = f2(horn1axis[i] + skreflengd * (s2 / 2), horn2axis[i] + skreflengd * (s2 / 2),
                    horn1hradiaxis[i], horn2hradiaxis[i])
            s4 = f2(horn1axis[i] + skreflengd * s3, horn2axis[i] + skreflengd * s3, horn1hradiaxis[i],
                                 horn2hradiaxis[i])
            w = horn2hradiaxis[i] + (s1 + s2 * 2 + s3 * 2 + s4) / 6 * skreflengd
            horn2hradiaxis.append(w)
            horn2axis.append((horn2axis[i] + skreflengd * w) % (math.pi*2))
        if sp9:
            return horn1axis, horn2axis, horn1hradiaxis, horn2hradiaxis
        else:
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
        return math.sin(theta) * fasti

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

        #k1 = m2 * l1 * math.pow(omega1, 2) * math.sin(d) * math.cos(d)
        #k2 = m2 * g * math.sin(theta2) * math.cos(d)
        #k3 = m2 * l2 * math.pow(omega2, 2) * math.sin(d)
        #k4 = (m1 + m2) * g * math.sin(theta1)
        #k5 = (m1 + m2) * l1 - m2 * l1 * math.pow(math.cos(d), 2)

        k1 = m2*l2*math.pow(omega2,2)*math.sin(d)
        k2 =-((m1+m2)*g*math.sin(theta1))
        k3 = m2*l1*math.pow(omega1,2)*math.sin(d)*math.cos(d)
        k4 = -m2*g*math.sin(theta2)*math.cos(d)
        k5 = ((m1+m2)*l1 - m2*l1*math.cos(d)*math.cos(d))

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

        #k1 = m2 * l2 * math.pow(omega2, 2) * math.sin(d) * math.cos(d)
        #k2 = (m1 + m2) * ( g * math.sin(theta1) * math.cos(d))
        #k3 = l1 * math.pow(omega1, 2) * math.sin(d)
        #k4 = g * math.sin(theta2)
        #k5 = (m1 + m2) * l2 - m2 * l2 * math.pow(math.cos(d), 2)

        k1 = -m2*l2*math.pow(omega2,2)*math.sin(d)*math.cos(d)
        k2 = -((m1+m2)*g*math.sin(theta1))*math.cos(d)
        k3 = -(m1+m2)*l1*math.pow(omega1,2)*math.sin(d)
        k4 = -(m1+m2)*g*math.sin(theta2)
        k5 = (m2*l2*math.cos(d)*math.cos(d) - (m1+m2)*l2)

        if l2 == 0 or k5 == 0 or (k1 + k2 + k3 - k4) == 0:
            return 0

        theta2_2prime = (-k1 + k2 - k3 - k4) / k5
        return theta2_2prime
    def hornTohnit(self, th, L_1=None):
        if L_1 is None:
            L_1 = self.L_1
        return L_1 * math.sin(th), -L_1 * math.cos(th)

    def hornTohnitjunior(self, th, th2, L_1=None, L_2=None):
        if L_1 is None:
            L_1=self.L_1
        if L_2 is None:
            L_2=self.L_2
        return L_1 * math.sin(th) + L_2 * math.sin(th2), -L_1 * math.cos(th) - L_2 * math.cos(th2)


