import numpy as np
import math
import matplotlib.pyplot as plt
import random
from newton import Newton
from scipy import stats as stats

system = np.array([[15600, 7540, 20140, 0.07074],
                   [18760, 2750, 18610, 0.07220],
                   [17610, 14630, 13480, 0.07690],
                   [19170, 610, 18390, 0.07242]])

new_sat_pos = np.array([(np.pi / 8, -np.pi / 4),  # φ, θ eða phi, theta
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

def point_diff(A,B):
    return np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2)
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
    n = Newton(sys)
    ax.scatter(xhnit, yhnit, zhnit, c='blue', alpha=0.3)
    tolerance = 0.01
    ax.set_title('3D line plot geeks for geeks')

    for x in sys:
        ax.scatter(x[0], x[1], x[2], c='yellow', s=300)

    ax.set_proj_type('ortho')
    ax.set_box_aspect((1, 1, 1))
    plt.show()

def nyttSatPos(pol=0):

    nytt_loc = coords(math.pi*random.random(), random.random()*10000, constaltitude)
    if pol==1:
        return np.array([math.pi*random.random(), random.random()*10000, constaltitude])
    global sat_teljari
    sat_teljari = sat_teljari + 1
    print("Gervihnöttur númer " + str(sat_teljari) + " : " + str(nytt_loc))
    return nytt_loc

def spurning1(plot=True):
    n = Newton(system)
    svar = n.GaussNewton(x0, tolerance)
    print("---- svar 1 ----- :")
    print("X: " + '%.6f' % svar[0] + " Y: " + '%.6f' % svar[1] + " Z: " + '%.6f' % svar[2] + " d: " + '%.6f' % svar[3])
def spurning2(plot=True):
    svar_coords = coords(0, 0)
    print("---- svar 2 ----- :")
    print(f"A: {svar_coords[0]:.02f}, B: {svar_coords[1]:.02f}, C: {svar_coords[2]:.02f}, t: {svar_coords[3]:.02f}, d: {svar_coords[4]:.02f}")
def spurning3(plot=True):
    print("---- svar 3 ----- :")

    new_system = np.array([coords(*sat)[:-1] for sat in new_sat_pos])
    new_system_plus_skekkja = np.array(
        [coords(sat[0] + skekkja, sat[1])[:-1] if index < 2 else coords(sat[0] - skekkja, sat[1])[:-1] for index, sat in
         enumerate(new_sat_pos)])

    # setja réttan tíma á skekkjukerfið
    for index, sat_pos in enumerate(new_system):
        new_system_plus_skekkja[index][-1] = sat_pos[-1]

    n3 = Newton(new_system)
    svaran = n3.GaussNewton(x0, tolerance)
    print("lausnin án skekkju   X: " + '%.6f' % svaran[0] + " Y: " + '%.6f' % svaran[1] + " Z: " + '%.6f' % svaran[
        2] + " d: " + '%.6f' % svaran[3])

    n2 = Newton(new_system_plus_skekkja)
    svarmed = n2.GaussNewton(x0, tolerance)
    print("lausnin með skekkju  X: " + '%.6f' % svarmed[0] + " Y: " + '%.6f' % svarmed[1] + " Z: " + '%.6f' % svarmed[
        2] + " d: " + '%.6f' % svarmed[3])
    print("Skekkjan sjálf : " + '%.6f' % point_diff(svaran, svarmed) + " kílómetrar")
def spurning4(plot=True):
    print("---- svar 4 ----- :")
    skekkja = 1e-8
    upphafsgildi = np.array([0, 0, 6370, 0])
    list_of_positions = []
    new_system = np.array([coords(*sat)[:-1] for sat in new_sat_pos])
    for i in range(16):
        new_system_with_error = np.array(
            [coords(sat[0] + skekkja, sat[1])[:-1] if i & (1 << index) else coords(sat[0] - skekkja, sat[1])[:-1]
             for index, sat in enumerate(new_sat_pos)])
        # uncomment to verify if correct:
        # new_systems_with_error = np.array([1 if (i & (1<<index))  else 0 for index, sat in enumerate(new_sat_pos)])
        # print(new_systems_with_error)
        for index, sat_pos in enumerate(new_system):
            new_system_with_error[index][-1] = sat_pos[-1]
        n3 = Newton(new_system_with_error)
        list_of_positions.append(n3.GaussNewton(upphafsgildi, tolerance))
    villu_positions = []
    for index, position in enumerate(list_of_positions):
        villu_positions.append(position)
        # villa_new = abs(position[0]) + abs(position[1]) + abs((position[2] - earthaltitude)) + abs(position[3])
    # plotta upp
    #plt.scatter([i for i in range(16)], [point_diff(x0, position) for position in list_of_positions])

    print(f"max: {max([point_diff(upphafsgildi, position) for position in list_of_positions]) * 1000:.04f}m")
    print(f"min: {min([point_diff(upphafsgildi, position) for position in list_of_positions]) * 1000:.04f}m")
    if plot:
        A_dreifing = [position[0] for position in list_of_positions]
        B_dreifing = [position[1] for position in list_of_positions]
        C_dreifing = [position[2] - earthaltitude for position in list_of_positions]
        plt.plot(A_dreifing)
        plt.plot(B_dreifing)
        plt.plot(C_dreifing)
        plt.show()
def spurning5(plot=True):
    # ------------- 5 ---------------


    print("---- svar 5 ----- :")
    new_sat_pos = np.array([[np.pi / 2, np.pi / 2],  # φ, θ, phi, theta
                            [np.pi / 2, np.pi / 2],
                            [np.pi / 2, np.pi / 2],
                            [np.pi / 2, np.pi / 2],])

    '''
    skekkja5 = 0.1
    for i in range(4):
        new_sat_pos[i][0] += (random.random()-.5) * skekkja5
        new_sat_pos[i][1] += (random.random()-.5) * skekkja5
    print(new_sat_pos)
    '''

    # skekkja = 0.1
    '''
    new_sat_pos = np.array([ [[1.55285912 1.599031 ],
                             [1.53712495 1.62040946],
                             [1.57151953 1.61481681],
                             [1.56491249 1.53779567]],])
    '''
    # skekkja = 0.01
    new_sat_pos = np.array([ [1.57098865 ,1.57282701],
                             [1.57508225 ,1.57233899],
                             [1.5707446  ,1.56733488],
                             [1.56586073 ,1.56823521],])

    n5system = [coords(phi, theta)[:-1] for phi, theta in new_sat_pos]

    n5 = Newton(n5system)
    if plot:
        plot3d(n5.system)
    print(n5.GaussNewton(x0, tolerance))
    print(point_diff(x0,n5.GaussNewton(x0, tolerance)))


# random sat positions hér svo hægt sé að nota í lið 7 og bera saman við lið 6
random_sat_positions = np.array([[nyttSatPos(1) for _ in range(4)] for _ in range(8)])


def spurning6(plot=True):
    print("---- svar 6 ----- :")
    skekkjusafn = []


    for oft in range(0,4):
        new_skekkja = (oft + 1) * skekkja
        #new_sat_pos = np.array([nyttSatPos(1),nyttSatPos(1),nyttSatPos(1),nyttSatPos(1)])
        new_sat_pos = random_sat_positions[oft]
        new_system = np.array([coords(*sat)[:-1] for sat in new_sat_pos])
        for i in range(16):
            new_system_with_error = np.array([coords(sat[0] + new_skekkja, sat[1])[:-1] if i & (1 << index) else coords(sat[0] - new_skekkja, sat[1])[:-1] for index, sat in enumerate(new_sat_pos)])
            for index, sat_pos in enumerate(new_system):
                new_system_with_error[index][-1] = sat_pos[-1]
            n3 = Newton(new_system_with_error)
            # print(x0)
            print(n3.GaussNewton(x0, tolerance))
            skekkjusafn.append(point_diff(x0, n3.GaussNewton(x0, tolerance)))
            print(str(oft) + " , " + str(i))
    # repeat with new error
    for oft in range(4,8):
        new_skekkja = (-oft - 1) * skekkja
        #new_sat_pos = np.array([nyttSatPos(1),nyttSatPos(1),nyttSatPos(1),nyttSatPos(1)])
        new_sat_pos = random_sat_positions[oft]
        new_system = np.array([coords(*sat)[:-1] for sat in new_sat_pos])
        for i in range(16):
            new_system_with_error = np.array([coords(sat[0] + new_skekkja, sat[1])[:-1] if i & (1 << index) else coords(sat[0] - new_skekkja, sat[1])[:-1] for index, sat in enumerate(new_sat_pos)])
            for index, sat_pos in enumerate(new_system):
                new_system_with_error[index][-1] = sat_pos[-1]
            n3 = Newton(new_system_with_error)
            # print(x0)
            print(n3.GaussNewton(x0, tolerance))
            skekkjusafn.append(point_diff(x0, n3.GaussNewton(x0, tolerance))*1000)  # breyta í metra
            print(str(oft) + " , " + str(i))
        print(f"number of skekkjur: {len(skekkjusafn)}")
    if plot:
        plt.style.use('fivethirtyeight')
        plt.hist(skekkjusafn, bins=20, edgecolor='black', density=True)
        plt.title('Means')
        plt.xlabel('bins')
        plt.ylabel('values')
        #plt.tight_layout()
        #x = np.arange(0, 1, 0.0001)
        #x1 = stats.norm.pdf(x, 0.5, 1 / math.sqrt(12 * len(skekkjusafn)))
        #plt.plot(x, x1, linewidth=1, color="black")
        plt.show()

def spurning7(plot=True):
    print("---- svar 7 ----- :")
def spurning8(plot=True):
    print("---- svar 8 ----- :")
def spurning9(plot=True):
    print("---- svar 9 ----- :")

if __name__ == '__main__':
    spurningarlisti = [
    spurning1,
    spurning2,
    spurning3,
    spurning4,
    spurning5,
    spurning6,
    spurning7,
    spurning8,
    spurning9,
    ]

    hvada_spurningar_a_ad_keyra = [6]  # <<<
    plot = True
    for spurningu in hvada_spurningar_a_ad_keyra:
        spurningarlisti[spurningu-1](plot=plot)

    #plot3d(new_system)


