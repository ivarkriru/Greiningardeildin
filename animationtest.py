from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import statistics

import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time
from scipy import stats as stats

system = np.array([[15600, 7540, 20140, 0.07074],
                   [18760, 2750, 18610, 0.07220],
                   [17610, 14630, 13480, 0.07690],
                   [19170, 610, 18390, 0.07242]])

sp3_initial_sat = np.array([(np.pi / 8, -np.pi / 4),  # φ, θ eða phi, theta
                            (np.pi / 6, np.pi / 2),
                            (3 * np.pi / 8, 2 * np.pi / 3),
                            (np.pi / 4, np.pi / 6),
                            ])
c = 299792.458
constaltitude = 26570
earthaltitude = 6370
tolerance = 0.0001
x0 = np.array([0, 0, 6370, 0])
sat_teljari = 0
skekkja = 1e-8
satkerfi_fjoldi = 9
sample_fjoldi = 100

constaltitude = 26570
earthaltitude = 6370

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
N=1000

def gen(n,phi=0,theta=0,hlutfall = 1):
    phizero = 0
    while phizero < 2*np.pi:
        yield np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        phizero += 2*np.pi/n
        phi += 2*np.pi/n

def coords(phi, theta, altitude=constaltitude):
    if 0 <= phi <= math.pi:
        A = altitude * np.sin(phi) * np.cos(theta)
        B = altitude * np.sin(phi) * np.sin(theta)
        C = altitude * np.cos(phi)
        # distance = numpy.sqrt(numpy.power((A-0),2)+numpy.power((B-0),2)+numpy.power((C-6370),2))
        distance = np.sqrt((A - 0) ** 2 + (B - 0) ** 2 + (C - earthaltitude) ** 2)
        time = distance / c

        return [A, B, C, time, distance]
    else:
        return "incorrect values"

def update(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])
def create_animation(data: list):

    # Setting the axes properties

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #line2, = ax.plot(data2[0, 0:1], data2[1, 0:1], data2[2, 0:1])
    ani_list = [animation.FuncAnimation(fig, update, N, fargs=(data_, ax.plot(np.array(data_)[0, 0:1], np.array(data_)[1, 0:1], np.array(data_)[2, 0:1])[0]), interval=10/N, blit=False) for data_ in data]
    #ani_list = [animation.FuncAnimation(fig, update, N, fargs=(data_, line_), interval=10/N, blit=False) for data_, line_ in data]

    #ani.save('matplot003.gif', writer='imagemagick')

    xhnit = []
    yhnit = []
    zhnit = []

    takmark = 300
    for x in range(0, takmark):
        svar = coords((x * 113) % (math.pi), (x * 7) % math.pi * 2, earthaltitude)
        xhnit.append(svar[0])
        yhnit.append(svar[1])
        zhnit.append(svar[2])

    ax.scatter(xhnit, yhnit, zhnit, c='blue', alpha=0.3)

    ax.set_xlim(-constaltitude, constaltitude)
    ax.set_ylim(-constaltitude, constaltitude)
    ax.set_zlim(-constaltitude, constaltitude)
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1, 1, 1))
    plt.show()

if __name__ == '__main__':
    N = 1000
    data = np.array(list(gen(N,np.pi,np.pi/2,0.4))).T
    data = data * constaltitude

    data2 = np.array(list(gen(N,5,12,0.1))).T
    data2 = data2 * constaltitude


    data3 = np.array(list(gen(N,3,9,2))).T
    data3 = data3 * constaltitude


    data4 = np.array(list(gen(N,20,1))).T
    data4 = data4 * constaltitude
    create_animation([data, data2, data3, data4])
    print([data, data2, data3])
