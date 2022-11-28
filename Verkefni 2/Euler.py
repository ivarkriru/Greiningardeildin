import numpy as np
y = lambda t, y : t * y - np.power(y, 3)

def euler(y0,n,T):
    h = T/n
    t = 0
