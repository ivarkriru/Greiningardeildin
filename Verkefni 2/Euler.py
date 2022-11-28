import numpy as np
import matplotlib.pyplot as plt
g = 9.81
L =2
f2 = lambda t, yinner: np.power(t, 2) +2
#f = lambda t, yinner: t * yinner - np.power(t, 3)


f = lambda yinner: np.array(yinner, (-g/L * np.sin(yinner)))

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
        yaxis.append(yaxis[i-1] + f(yaxis[i-1]) * h)
    print(xaxis)
    print(yaxis)
    plt.plot(xaxis, yaxis)
#    np.linspace(0,5)


#euler(0,5,3)
euler(0.1,20,3)
#euler(0,500,3)
plt.show()