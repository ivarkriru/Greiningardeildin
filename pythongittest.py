import numpy as np
from numpy import linalg as LA

x_fixed = 2

def bisection(f,a,b,tol):
    if f(x_fixed,a)*f(x_fixed,b) >= 0:
        print("Bisection method fails.")
        return None
    else:
        fa=f(x_fixed,a)
        while (b-a)/2>tol:
            c=(a+b)/2
            fc=f(x_fixed,c)
            if fc==0:break
            if fc*fa<0:
                b=c
            else:
                a=c
                fa=fc
    return((a+b)/2)

#def f(y):
#    return pow(x_fixed,3) - 3*pow(x_fixed,2)*pow(y,2) + x_fixed*y +1
f = lambda x, y: pow(x,3) - 3*pow(x,2)*pow(y,2) + x*y + 1

if __name__ == '__main__':
    tolerance = 0.01
    print(bisection(f,-4,0,tolerance))
