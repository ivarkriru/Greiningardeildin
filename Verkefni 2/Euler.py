import numpy as np
import matplotlib.pyplot as plt
f = lambda t, yinner: np.power(t, 2) +2

def euler(y0,n,T):
    h = T/n
    t = 0
    xaxis = []
    yaxis = []
    xaxis.append(t)
    yaxis.append(y0)
    y=y0
    for i in range(1, n):
        t = t + h
        xaxis.append(t)
        yaxis.append(yaxis[i-1] + f(t,yaxis[i-1]) + h)

    x = range(y0,T)
    y=f(x,x)
    plt.plot(x,y)
    plt.plot(xaxis,yaxis)
    np.linspace(0,5)
    plt.show()


euler(0,5,3)

euler(0,20,3)
euler(0,200,3)