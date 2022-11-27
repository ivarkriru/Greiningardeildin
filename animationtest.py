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

def turner(data,alpha,beta,epsilon):
    turnmatrix = np.array([[np.cos(alpha)*np.cos(beta),np.cos(alpha)*np.sin(beta)*np.sin(epsilon)-np.sin(alpha)*np.cos(epsilon),np.cos(alpha)*np.sin(beta)*np.cos(epsilon)+np.sin(alpha)*np.sin(epsilon)],
                           [np.sin(alpha)*np.cos(beta),np.sin(alpha)*np.sin(beta)*np.sin(epsilon)+np.cos(alpha)*np.cos(epsilon),np.sin(alpha)*np.sin(beta)*np.cos(epsilon)-np.cos(alpha)*np.sin(epsilon)],
                           [-np.sin(beta),np.cos(beta)*np.sin(epsilon),np.cos(beta)*np.cos(epsilon)]])
    data = np.transpose(data)
    return np.transpose(np.matmul(data,turnmatrix))

N = 100


data = np.array(list(gen(N,random.random()*100,random.random()*100,random.random()*100))).T
data = data * constaltitude
data = turner(data,random.random()*100,random.random()*100,random.random()*100)
line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])


data2 = np.array(list(gen(N,random.random()*100,random.random()*100,random.random()*100))).T
data2 = data2 * constaltitude
data2 = turner(data2,random.random()*100,random.random()*100,random.random()*100)
line2, = ax.plot(data2[0, 0:1], data2[1, 0:1], data2[2, 0:1])


data3 = np.array(list(gen(N,random.random()*100,random.random()*100,random.random()*100))).T
data3 = data3 * constaltitude
data3 = turner(data3,random.random()*100,random.random()*100,random.random()*100)
line3, = ax.plot(data3[0, 0:1], data3[1, 0:1], data3[2, 0:1])


data4 = np.array(list(gen(N,random.random()*100,random.random()*100,random.random()*100))).T
data4 = data4 * constaltitude
data4 = turner(data4,random.random()*100,random.random()*100,random.random()*100)
line4, = ax.plot(data4[0, 0:1], data4[1, 0:1], data4[2, 0:1])

# Setting the axes properties

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=10000/N, blit=False)
ani2 = animation.FuncAnimation(fig, update, N, fargs=(data2, line2), interval=10000/N, blit=False)
ani3 = animation.FuncAnimation(fig, update, N, fargs=(data3, line3), interval=10000/N, blit=False)
ani4 = animation.FuncAnimation(fig, update, N, fargs=(data4, line4), interval=10000/N, blit=False)

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