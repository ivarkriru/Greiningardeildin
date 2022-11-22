import numpy
import numpy as np
import math
import numpy.linalg as la
import matplotlib.pyplot as plt


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
        if counter >=15 :
            print("------------reiknaði of lengi------------")
            break
    return (x)

def coords(theta,phi):
    if -math.pi/2<=phi<= 2 * math.pi and 0 <= theta <= math.pi*2:

        A = constaltitude*math.sin(phi)*math.cos(theta)
        B = constaltitude*math.sin(phi)*math.sin(theta)
        C = constaltitude*math.cos(phi)
        distance = numpy.sqrt(numpy.power((A-0),2)+numpy.power((B-0),2)+numpy.power((C-6370),2))
        time = distance/c

        return [A,B,C,distance,time]
    else:
        return "incorrect values"

def plot3d():
    fig = plt.figure()

    # syntax for 3-D projection
    ax = plt.axes(projection='3d')
    xhnit = []
    yhnit = []
    zhnit = []

    # defining all 3 axes
    takmark = 1000
    for x in range(0,takmark):
        svar = coords((x*7)%math.pi*2,(x*113)%(math.pi))
        xhnit.append(svar[0])
        yhnit.append(svar[1])
        zhnit.append(svar[2])

    # plotting

    ax.set_box_aspect((1,1,1))
    ax.scatter(xhnit, yhnit, zhnit, 'green')

    x0 = vigur
    tolerance = 0.01
    svar = newtonmult(x0, tolerance)
    ax.scatter(svar[0], svar[1], svar[2], 'red')


    ax.scatter(x0[0], x0[1], x0[2], 'purple')
    ax.set_title('3D line plot geeks for geeks')
    plt.show()
def skekkja():
    i1rangt = coords(math.pi/8, -math.pi/4)
    i2rangt = coords(math.pi/6, math.pi/2)
    i3rangt = coords(3 * math.pi / 8, 2 * math.pi / 3)
    i4rangt = coords(math.pi / 4, math.pi / 6)
    i1rett = coords((math.pi / 8) + 0.000000001, -math.pi / 4)
    i2rett = coords((math.pi / 6) + 0.000000001, math.pi / 2)
    i3rett = coords((3 * math.pi / 8) + 0.000000001, 2 * math.pi / 3)
    i4rett = coords((math.pi / 4) + 0.000000001, math.pi / 6)


if __name__ == '__main__':
    x0 = vigur
    tolerance = 0.01
    svar = newtonmult(x0, tolerance)
    print("X: " + '%.6f' % svar[0] + " Y: " + '%.6f' % svar[1] + " Z: " + '%.6f' % svar[2] + " d: " + '%.6f' % svar[3])
    svar_coords = coords(0,0)
    print("A: " + '%.6f' % svar_coords[0] + " B: " + '%.6f' % svar_coords[1] + " C: " + '%.6f' % svar_coords[2] + " d: " + '%.6f' % svar_coords[3] + " t: " + '%.6f' % svar_coords[4])
    plot3d()
    skekkja()