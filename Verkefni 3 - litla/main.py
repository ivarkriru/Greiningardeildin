import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as la
import random

def solutionfall(x,c1=1,c2=1):
    return c1*math.exp(x) + c2*math.exp(-x)

def solutionfalldiff(x,c1=1,c2=1):
    return c1*math.exp(x) - c2*math.exp(-x)

def tvimerkt(f,x,h):
    k1 = f(x+h)
    k2 = -2*f(x)
    k3 = f(x-h)
    k4 = (math.pow(h,2))
    k5 = (k1+k2+k3)/k4
    return k5

def merkt(f,naesti,undan,h):
    return (f(naesti)-f(undan))/(2*h)

def daemi1():
    n = 100
    c1 = 1
    c2 = 1
    bil = np.array([*range(0, n)]) * 1 / (n - 1)
    h = bil[1]

    solutiondata = []
    for x in bil:
        solutiondata.append(solutionfall(x,c1=c1,c2=c2))
    A = np.zeros([n - 1, n - 1])

    b = np.zeros([n - 1, 1])

    gildi = [1 / (h * h), -2 / (h * h), 1 / (h * h)]

    for index, x in enumerate(bil[1:-1]):
        for j in range(3):
            if not (j + index > n - 3):
                A[index, j + index - 1] = gildi[j]
        b[index, 0] = 0

    b[0, 0] = 2
    b[-1, 0] = 3.0861

    A[0, 0] = 1
    A[0, 1] = 0

    A[0, -1] = 0
    A[-1, -1] = 1
    A[-2, -1] = gildi[2]
    A[-3, -2] = gildi[2]
    A[-2, -2] = gildi[1]
    svar = la.solve(A, b)

    plt.plot(bil,solutiondata,c="green")
    plt.plot(bil[1:],svar,c="red")

    '''
    skekkja = []
    prosentuskekkja = []
    for index, x in enumerate(solutiondata[1:-1]):
        skekkja.append((x - A[index]))
        prosentuskekkja.append((x - A[index])/x)
    print(sum(skekkja))
    plt.plot(bil[1:-1], skekkja)
    #plt.plot(bil[1:-1], prosentuskekkja)
    '''
    plt.show()

def daemi2():
    n = 100
    c1 = -0.5820
    c2 = 1.5820
    bil = np.array([*range(0, n)]) * 1 / (n - 1)
    h = bil[1]

    solutiondata = []
    for x in bil:
        solutiondata.append(solutionfall(x,c1=c1,c2=c2))

    A = np.zeros([n - 1, n - 1])

    b = np.zeros([n - 1, 1])

    gildi = [1/(h*h), -2/(h*h), 1/(h*h)]

    for index, x in enumerate(bil[1:-1]):
        for j in range(3):
            if not (j + index > n - 3):
                A[index, j + index - 1] = gildi[j]
        b[index, 0] = 0

    b[0, 0] = 1
    b[-1, 0] = -1

    A[0, 0] = 1
    A[0, 1] = 0

    A[-1, -1] = 1

    A[-2, -1] = gildi[2]
    A[-3, -2] = gildi[2]
    A[-2, -2] = gildi[1]
    A[0, -1] = 0
    svar = la.solve(A, b)

    plt.plot(bil, solutiondata, c="green")
    plt.plot(bil[1:], svar, c="red")


    plt.show()

def daemi3():
    n = 100
    c1 = 0.4255
    c2 = 0.4255
    bil = np.array([*range(0, n)]) * 1 / (n - 1)
    h = bil[1]

    solutiondata = []
    for x in bil:
        solutiondata.append(solutionfall(x, c1=c1, c2=c2))

    A = np.zeros([n - 1, n - 1])

    b = np.zeros([n - 1, 1])

    gildi = [1 / (h * h), -2 / (h * h), 1 / (h * h)]

    for index, x in enumerate(bil[1:-1]):
        for j in range(3):
            if not (j + index > n - 3):
                A[index, j + index - 1] = gildi[j]
        b[index, 0] = 0

    b[0, 0] = 0
    b[-1, 0] = 1

    A[0, 0] = -3/(2*h)
    A[0, 1] = 4/(2*h)
    A[0, 2] = -1/(h)

    A[-1, -1] = -3/(-2*h)
    A[-1, -2] = 4/(-2*h)
    A[-1, -3] = -1/(-h)

    A[-2, -1] = gildi[2]
    A[-3, -2] = gildi[2]
    A[-2, -2] = gildi[1]
    A[0, -1] = 0

    svar = la.solve(A, b)

    plt.plot(bil, solutiondata, c="green")
    plt.plot(bil[1:], svar, c="red")

    plt.show()
def daemi4():

    n = 1000
    c1 = 0
    c2 = -0.5
    bil = np.array([*range(0, n)]) * 1 / (n - 1)
    h = bil[1]

    solutiondata = []
    for x in bil:
        solutiondata.append(solutionfall(x, c1=c1, c2=c2))

    A = np.zeros([n - 1, n - 1])

    b = np.zeros([n - 1, 1])

    gildi = [1 / (h * h), -2 / (h * h), 1 / (h * h)]

    for index, x in enumerate(bil[1:-1]):
        for j in range(3):
            if not (j + index > n - 3):
                A[index, j + index - 1] = gildi[j]
        b[index, 0] = 0

    b[0, 0] = 1
    b[-1, 0] = 0

    A[0, 0] = -5 / (2 * h)
    A[0, 1] = 4 / (2 * h)
    A[0, 2] = -1 / (2 * h)

    A[-1, -1] = -1 / (-2 * h)
    A[-1, -2] = 4 / (-2 * h)
    A[-1, -3] = -1 / (-2 * h)

    A[-2, -1] = gildi[2]
    A[-3, -2] = gildi[2]
    A[-2, -2] = gildi[1]
    A[0, -1] = 0
    svar = la.solve(A, b)

    plt.plot(bil, solutiondata, c="green")
    plt.plot(bil[1:], svar, c="red")
    plt.show()

if __name__ == '__main__':
    #daemi1()
    #daemi2()
    #daemi3()
    daemi4()