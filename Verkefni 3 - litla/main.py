import numpy as np
import math
import matplotlib.pyplot as plt

def solutionfall(x,c1=1,c2=1):
    return c1*math.exp(x) + c2*math.exp(-x)

def tvimerkt(f,x,h):
    return (f(x-h)-2*f(x)+f(x+h))/(math.pow(h,2))


def daemi1():
    n = 100
    bil = np.array([*range(0, n)]) * 1 / (n - 1)
    data = []
    for x in bil:
        data.append(solutionfall(x))

    datahermi = []
    for x in bil:
        datahermi.append(tvimerkt(solutionfall, x, bil[1]))
    # plt.plot(bil,data,c="green")
    # plt.plot(bil,datahermi,c="red")

    skekkja = []
    for index, x in enumerate(data):
        skekkja.append(x - datahermi[index])

    plt.plot(bil, skekkja)
    plt.show()

def daemi2():
    n = 100
    bil = np.array([*range(0, n)]) * 1 / (n - 1)
    data = []
    for x in bil:
        data.append(solutionfall(x))

    datahermi = []
    for x in bil:
        datahermi.append(tvimerkt(solutionfall, x, bil[1]))
    # plt.plot(bil,data,c="green")
    # plt.plot(bil,datahermi,c="red")

    skekkja = []
    for index, x in enumerate(data):
        skekkja.append(x - datahermi[index])

    plt.plot(bil, skekkja)
    plt.show()

def daemi3():
    n = 100
    bil = np.array([*range(0, n)]) * 1 / (n - 1)
    data = []
    for x in bil:
        data.append(solutionfall(x))

    datahermi = []
    for x in bil:
        datahermi.append(tvimerkt(solutionfall, x, bil[1]))
    # plt.plot(bil,data,c="green")
    # plt.plot(bil,datahermi,c="red")

    skekkja = []
    for index, x in enumerate(data):
        skekkja.append(x - datahermi[index])

    plt.plot(bil, skekkja)
    plt.show()

def daemi4():
    n = 100
    bil = np.array([*range(0, n)]) * 1 / (n - 1)
    data = []
    for x in bil:
        data.append(solutionfall(x))

    datahermi = []
    for x in bil:
        datahermi.append(tvimerkt(solutionfall, x, bil[1]))
    # plt.plot(bil,data,c="green")
    # plt.plot(bil,datahermi,c="red")

    skekkja = []
    for index, x in enumerate(data):
        skekkja.append(x - datahermi[index])

    plt.plot(bil, skekkja)
    plt.show()

if __name__ == '__main__':
    daemi1()
    #daemi2()
    #daemi3()
    #daemi4()