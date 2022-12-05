import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as la

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
    n = 10000
    c1 = 1
    c2 = 1
    bil = np.array([*range(0, n)]) * 1 / (n - 1)
    h = bil[1]

    solutiondata = []
    for x in bil:
        solutiondata.append(solutionfall(x))

    A = np.zeros([n-2,n-2])
    b = np.zeros([n-2,1])
    gildi = [1/(h*h),-2/(h*h),1/(h*h)]

    for index,x in enumerate(bil[1:-1]):
        for j in range(3):
            if not (j + index > n - 3):
                A[index, j + index] = gildi[j]
        b[index,0] = c1*math.exp(x) + c2*math.exp(-x)
    #plt.plot(bil,solutiondata,c="green")
    #plt.plot(bil[1:-1],datahermi,c="red")
    svar = la.solve(A, b)
    plt.plot(bil,solutiondata,c="green")
    plt.plot(bil[1:-1],svar,c="red")


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
    n = 10
    bil = np.array([*range(0, n)]) * 1 / (n - 1)
    data = []
    for x in bil:
        data.append(solutionfall(x))

    datahermi = []
    for x in bil:
        datahermi.append(tvimerkt(solutionfall, x, bil[1]))
    plt.plot(bil,data,c="green")
    plt.plot(bil,datahermi,c="red")

    skekkja = []
    prosentuskekkja = []
    for index, x in enumerate(data):
        skekkja.append((x - datahermi[index]))
        prosentuskekkja.append((x - datahermi[index]) / x)

    #plt.plot(bil, skekkja)
    #plt.plot(bil, prosentuskekkja)
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