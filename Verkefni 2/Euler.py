import numpy as np
import matplotlib.pyplot as plt
f2 = lambda t, yinner: np.power(t, 2) +2
f = lambda t, yinner: t * yinner - np.power(t, 3)

def euler(y0,n,T):
    h = T/n
    t = 0
    xaxis = []
    yaxis = []
    xaxis.append(t)
    yaxis.append(y0)
    y = y0
    for i in range(1, n+1):
        t = t+h
        xaxis.append(t)
        yaxis.append(yaxis[i-1] + f(t, yaxis[i-1]) * h)

    plt.plot(xaxis, yaxis)
#    np.linspace(0,5)


euler(2,5,3)
euler(2,20,3)
euler(2,200,3)
fall(2,200,3)
plt.show()