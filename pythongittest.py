import numpy as np
from numpy import linalg as LA

x_fixed = 2

def bisection(a,b,tol):
    if f(a)*f(b) >= 0:
        print("Bisection method fails.")
        return None
    else:
        fa=f(a)
        while (b-a)/2>tol:
            c=(a+b)/2
            fc=f(c)
            if fc==0:break
            if fc*fa<0:
                b=c
            else:
                a=c
                fa=fc
    return((a+b)/2)

def f(y):
    return pow(x_fixed,3) - 3*pow(x_fixed,2)*pow(y,2) + x_fixed*y +1


if __name__ == '__main__':
    for x in range(-4,4):
        print(x)
        print(f(x))
    tolerance = 0.01
    print(bisection(-4,0,tolerance))
