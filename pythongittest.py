
import numpy as np
from numpy import linalg as LA

"""
 timadaemi 

 kerfið F(x) = (x^2-y^2
                x^3 + y^3)

 F'(x) = (2x    -2y
          3x^2  3y^2)

"""

def DF(x):
    return np.array([[2*pow(x[0],1) ,-2*pow(x[1],1)], [3*pow(x[0],2) , 3*pow(x[1],2)]])

def F(x):
    return np.array([pow(x[0],2) -pow(x[1],2), pow(x[0],3) + pow(x[1],3)])

def newtonmult(x0, tol):
    '''x0 er vigur i R^n skilgreindur t.d. sem
    x0=np.array([1,2,3])
    gert ráð fyrir að F(x) og Jacobi fylki DF(x) séu skilgreind annars staðar'''
    x = x0
    oldx = x + 2 * tol
    while LA.norm(x - oldx, np.inf) > tol:
        oldx = x
        s = -LA.solve(DF(x), F(x))
        x = x + s
    return (x)

if __name__ == '__main__':
    x0 = np.array([1,1])
    print(DF(x0))
    print(F(x0))
    tolerance = 0.0005
    print(newtonmult(x0,tolerance))
    print(F([0.00067664, 0.00067664]))
