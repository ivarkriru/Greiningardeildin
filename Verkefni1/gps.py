import random
import numpy as np
import math
import numpy.linalg as la
import matplotlib.pyplot as plt

system = np.array([[15600, 7540, 20140, 0.07074],
                   [18760, 2750, 18610, 0.07220],
                   [17610, 14630, 13480, 0.07690],
                   [19170, 610, 18390, 0.07242]])

new_sat_pos = np.array([(np.pi / 8, -np.pi / 4),  # φ, θ, phi, theta
                        (np.pi / 6, np.pi / 2),
                        (3 * np.pi / 8, 2 * np.pi / 3),
                        (np.pi / 4, np.pi / 6),
                        ])
c = 299792.458
constaltitude = 26570
earthaltitude = 6370
tolerance = 0.01
x0 = np.array([0, 0, 6370, 0])
sat_teljari = 0
skekkja = 1e-8

class newton:
    def __init__(self, system=system):
        self.system = system

    def fall(self, x):
        return np.array([self.f1(x[0], x[1], x[2], x[3]),
                         self.f2(x[0], x[1], x[2], x[3]),
                         self.f3(x[0], x[1], x[2], x[3]),
                         self.f4(x[0], x[1], x[2], x[3])])

    def f1(self, x, y, z, d):
        return pow((x - self.system[0][0]), 2) + pow((y - self.system[0][1]), 2) + pow((z - self.system[0][2]),
                                                                                       2) - pow(c, 2) * pow(
            (self.system[0][3] - d), 2)

    def f2(self, x, y, z, d):
        return pow((x - self.system[1][0]), 2) + pow((y - self.system[1][1]), 2) + pow((z - self.system[1][2]),
                                                                                       2) - pow(c, 2) * pow(
            (self.system[1][3] - d), 2)

    def f3(self, x, y, z, d):
        return pow((x - self.system[2][0]), 2) + pow((y - self.system[2][1]), 2) + pow((z - self.system[2][2]),
                                                                                       2) - pow(c, 2) * pow(
            (self.system[2][3] - d), 2)

    def f4(self, x, y, z, d):
        return pow((x - self.system[3][0]), 2) + pow((y - self.system[3][1]), 2) + pow((z - self.system[3][2]),
                                                                                       2) - pow(c, 2) * pow(
            (self.system[3][3] - d), 2)

    def dF(self, vigur):
        return np.array([[2 * vigur[0] - 2 * self.system[0][0], 2 * vigur[1] - 2 * self.system[0][1],
                          2 * vigur[2] - 2 * self.system[0][2],
                          2 * self.system[0][3] * pow(c, 2) - 2 * pow(c, 2) * vigur[3]],
                         [2 * vigur[0] - 2 * self.system[1][0], 2 * vigur[1] - 2 * self.system[1][1],
                          2 * vigur[2] - 2 * self.system[1][2],
                          2 * self.system[1][3] * pow(c, 2) - 2 * pow(c, 2) * vigur[3]],
                         [2 * vigur[0] - 2 * self.system[2][0], 2 * vigur[1] - 2 * self.system[2][1],
                          2 * vigur[2] - 2 * self.system[2][2],
                          2 * self.system[2][3] * pow(c, 2) - 2 * pow(c, 2) * vigur[3]],
                         [2 * vigur[0] - 2 * self.system[3][0], 2 * vigur[1] - 2 * self.system[3][1],
                          2 * vigur[2] - 2 * self.system[3][2],
                          2 * self.system[3][3] * pow(c, 2) - 2 * pow(c, 2) * vigur[3]]])

    def GaussNewton(self, x0, tol):
        '''x0 er vigur i R^n skilgreindur t.d. sem
        x0=np.array([1,2,3])
        gert ráð fyrir að F(x) og Jacobi fylki DF(x) séu skilgreind annars staðar'''
        x = x0
        oldx = x + 2 * tol
        counter = 0
        AT=np.transpose(self.dF(x0))
        while la.norm(x - oldx, np.inf) > tol:
            oldx = x
            s = -la.solve(np.matmul(AT,self.dF(x0)), np.matmul(AT,self.fall(x)))
            x = x + s
            counter += 1
            if counter >= 15:
                print("------------reiknaði of lengi------------")
                break
        return (x)

def point_diff(A,B):
    return np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2)

def coords(phi, theta, altitude=constaltitude + earthaltitude):
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

new_system = np.array([coords(*sat)[:-1] for sat in new_sat_pos])

def plot3d(sys):
    fig = plt.figure()

    # syntax for 3-D projection
    ax = plt.axes(projection='3d')
    xhnit = []
    yhnit = []
    zhnit = []

    # defining all 3 axes
    takmark = 300
    for x in range(0, takmark):
        svar = coords((x * 113) % (math.pi), (x * 7) % math.pi * 2, earthaltitude)
        xhnit.append(svar[0])
        yhnit.append(svar[1])
        zhnit.append(svar[2])

    # plotting
    n = newton(sys)
    ax.scatter(xhnit, yhnit, zhnit, c='blue', alpha=0.3)
    tolerance = 0.01
    svar = n.GaussNewton(x0, tolerance)
    ax.scatter(svar[0], svar[1], svar[2], c='red', s=300)

    ax.scatter(x0[0], x0[1], x0[2], c='yellow', s=300)
    ax.set_title('3D line plot geeks for geeks')

    for x in sys:
        ax.scatter(x[0], x[1], x[2], c='yellow', s=300)

    ax.set_proj_type('ortho')
    ax.set_box_aspect((1, 1, 1))
    plt.show()

def nyttSatPos():
    nytt_loc = coords(math.pi*random.random(), random.random()*10000, constaltitude)
    global sat_teljari
    sat_teljari = sat_teljari + 1
    print("Gervihnöttur númer " + str(sat_teljari) + " : " + str(nytt_loc))
    return nytt_loc

def spurning1():
    n = newton(system)
    svar = n.GaussNewton(x0, tolerance)
    print("---- svar 1 ----- :")
    print("X: " + '%.6f' % svar[0] + " Y: " + '%.6f' % svar[1] + " Z: " + '%.6f' % svar[2] + " d: " + '%.6f' % svar[3])

def spurning2():
    svar_coords = coords(0, 0)
    print("---- svar 2 ----- :")
    print(f"A: {svar_coords[0]:.02f}, B: {svar_coords[1]:.02f}, C: {svar_coords[2]:.02f}, t: {svar_coords[3]:.02f}, d: {svar_coords[4]:.02f}")

def spurning3():
    print("---- svar 3 ----- :")

    new_system_plus_skekkja = np.array(
        [coords(sat[0] + skekkja, sat[1])[:-1] if index < 2 else coords(sat[0] - skekkja, sat[1])[:-1] for index, sat in
         enumerate(new_sat_pos)])

    # setja réttan tíma á skekkjukerfið
    for index, sat_pos in enumerate(new_system):
        new_system_plus_skekkja[index][-1] = sat_pos[-1]

    n3 = newton(new_system)
    svaran = n3.GaussNewton(x0, tolerance)
    print("lausnin án skekkju   X: " + '%.6f' % svaran[0] + " Y: " + '%.6f' % svaran[1] + " Z: " + '%.6f' % svaran[
        2] + " d: " + '%.6f' % svaran[3])

    n2 = newton(new_system_plus_skekkja)
    svarmed = n2.GaussNewton(x0, tolerance)
    print("lausnin með skekkju  X: " + '%.6f' % svarmed[0] + " Y: " + '%.6f' % svarmed[1] + " Z: " + '%.6f' % svarmed[
        2] + " d: " + '%.6f' % svarmed[3])
    print("Skekkjan sjálf : " + '%.6f' % point_diff(svaran, svarmed) + " metrar")

def spurning4():
    print("---- svar 4 ----- :")
    skekkja = 1e-8
    upphafsgildi = np.array([0, 0, 6370, 0])
    if True:
        list_of_positions = []
        for i in range(16):
            new_system_with_error = np.array(
                [coords(sat[0] + skekkja, sat[1])[:-1] if i & (1 << index) else coords(sat[0] - skekkja, sat[1])[:-1]
                 for index, sat in enumerate(new_sat_pos)])
            # uncomment to verify if correct:
            # new_systems_with_error = np.array([1 if (i & (1<<index))  else 0 for index, sat in enumerate(new_sat_pos)])
            # print(new_systems_with_error)
            for index, sat_pos in enumerate(new_system):
                new_system_with_error[index][-1] = sat_pos[-1]
            n3 = newton(new_system_with_error)
            list_of_positions.append(n3.GaussNewton(upphafsgildi, tolerance))
        villu_positions = []
        for index, position in enumerate(list_of_positions):
            print(position)
            villu_positions.append(position)
            # villa_new = abs(position[0]) + abs(position[1]) + abs((position[2] - earthaltitude)) + abs(position[3])
    # plotta upp
    plt.scatter([i for i in range(16)], [point_diff(x0, position) for position in list_of_positions])

    print(f"max: {max([point_diff(x0, position) for position in list_of_positions]) * 1000:.04f}m")
    print(f"min: {min([point_diff(x0, position) for position in list_of_positions]) * 1000:.04f}m")

    A_dreifing = [position[0] for position in list_of_positions]
    B_dreifing = [position[1] for position in list_of_positions]
    C_dreifing = [position[2] - earthaltitude for position in list_of_positions]
    # plt.plot(A_dreifing)
    # plt.plot(B_dreifing)
    # plt.plot(C_dreifing)
    # plt.show()

def spurning5():
    # ------------- 5 ---------------

    new_sat_pos = np.array([[np.pi / 2, np.pi / 2],  # φ, θ, phi, theta
                            [np.pi / 2, np.pi / 2],
                            [np.pi / 2, np.pi / 2],
                            [np.pi / 2, np.pi / 2],
                            ])
    skekkja5 = 1e-5
    for i in range(4):
        new_sat_pos[i][0] + (random.randrange(-1, 1) * skekkja5)
        new_sat_pos[i][1] + (random.randrange(-1, 1) * skekkja5)
    n5 = newton()
def spurning6():
    pass
def spurning7():
    pass
def spurning8():
    pass
def spurning9():
    pass

if __name__ == '__main__':
    spurning1()
    spurning2()
    spurning3()
    spurning4()
    spurning5()
    spurning6()
    spurning7()
    spurning8()
    spurning9()

    plot3d(new_system)


