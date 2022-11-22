import numpy
import numpy as np
import math
import numpy.linalg as la
system = np.array([  [15600, 7540, 20140, 0.07074],
                     [18760, 2750, 18610, 0.07220],
                     [17610, 14630, 13480, 0.07690],
                     [19170, 610, 18390, 0.07242]])
c = 299792.458
constaltitude = 26570
tolerance = 0.01
vigur = np.array([0, 0, 6370, 0])

def fall(x):
    return np.array([f1(x[0],x[1],x[2],x[3]),
                     f2(x[0],x[1],x[2],x[3]),
                     f3(x[0],x[1],x[2],x[3]),
                     f4(x[0],x[1],x[2],x[3])])

def f1(x,y,z,d):
    return pow((x - system[0][0]),2) + pow((y - system[0][1]),2) + pow((z - system[0][2]),2) - pow(c,2)*pow((system[0][3]-d),2)

def f2(x, y, z, d):
    return pow((x - system[1][0]),2) + pow((y - system[1][1]),2) + pow((z - system[1][2]),2) - pow(c,2)*pow((system[1][3]-d),2)

def f3(x, y, z, d):
    return pow((x - system[2][0]),2) + pow((y - system[2][1]),2) + pow((z - system[2][2]),2) - pow(c,2)*pow((system[2][3]-d),2)

def f4(x, y, z, d):
    return pow((x - system[3][0]),2) + pow((y - system[3][1]),2) + pow((z - system[3][2]),2) - pow(c,2)*pow((system[3][3]-d),2)


def dF(vigur):
    return np.array([[2*vigur[0]-2*system[0][0], 2*vigur[1]-2*system[0][1], 2*vigur[2]-2*system[0][2], 2*system[0][3]*pow(c,2)- 2*pow(c,2)*vigur[3]],
                     [2*vigur[0]-2*system[1][0], 2*vigur[1]-2*system[1][1], 2*vigur[2]-2*system[1][2], 2*system[1][3]*pow(c,2)- 2*pow(c,2)*vigur[3]],
                     [2*vigur[0]-2*system[2][0], 2*vigur[1]-2*system[2][1], 2*vigur[2]-2*system[2][2], 2*system[2][3]*pow(c,2)- 2*pow(c,2)*vigur[3]],
                     [2*vigur[0]-2*system[3][0], 2*vigur[1]-2*system[3][1], 2*vigur[2]-2*system[3][2], 2*system[3][3]*pow(c,2)- 2*pow(c,2)*vigur[3]]])

def newtonmult(x0, tol):
    '''x0 er vigur i R^n skilgreindur t.d. sem
    x0=np.array([1,2,3])
    gert ráð fyrir að F(x) og Jacobi fylki DF(x) séu skilgreind annars staðar'''
    x = x0
    oldx = x + 2 * tol
    counter = 0
    while la.norm(x - oldx, np.inf) > tol:
        oldx = x
        s = -la.solve(dF(x), fall(x))
        x = x + s
        counter +=1
        if counter >=10 :
            print("------------reiknaði of lengi------------")
            break
    return (x)

def coords(theta,phi):
    if 0<=phi<=math.pi/2 and 0<=theta<=math.pi*2:

        A = constaltitude*math.sin(phi)*math.cos(theta)
        B = constaltitude*math.sin(phi)*math.sin(theta)
        C = constaltitude*math.cos(phi)
        distance = numpy.sqrt(numpy.power((A-0),2)+numpy.power((B-0),2)+numpy.power((C-6370),2))
        time = distance/c

        return [A,B,C,distance,time]
    else:
        return "incorrect values"

if __name__ == '__main__':
    x0 = vigur
    tolerance = 0.01
    svar = newtonmult(x0, tolerance)
    print("X: " + '%.6f' % svar[0] + " Y: " + '%.6f' % svar[1] + " Z: " + '%.6f' % svar[2] + " d: " + '%.6f' % svar[3])

    print(coords(0,0))