import statistics

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

def plot3d(sys,halfur=0):
    if halfur != 1:
        halfur = 0.5
    fig = plt.figure()

    # syntax for 3-D projection
    ax = plt.axes(projection='3d')
    xhnit = []
    yhnit = []
    zhnit = []

    # defining all 3 axes
    takmark = 300
    for x in range(0, takmark):
        svar = coords((x * 113) % (math.pi*halfur), (x * 7) % math.pi * 2, earthaltitude)
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
    ax.set_xlim(-constaltitude, constaltitude)
    ax.set_ylim(-constaltitude, constaltitude)
    ax.set_zlim(-constaltitude, constaltitude)
    ax.set_proj_type('ortho')
    ax.set_box_aspect((1, 1, 1))
    plt.show()

def nyttSatPos(pol=0,halfur=0):
    if halfur != 1:
        halfur = 0.5
    if pol==1:
        return np.array([math.pi*halfur*random.random(), random.random()*10000, constaltitude])
    global sat_teljari
    sat_teljari = sat_teljari + 1
    nytt_loc = coords(math.pi*halfur * random.random(), random.random() * 10000, constaltitude)
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

    new_system = np.array([coords(*sat)[:-1] for sat in sp3_initial_sat])
    new_system_plus_skekkja = np.array(
        [coords(sat[0] + skekkja, sat[1])[:-1] if index < 2 else coords(sat[0] - skekkja, sat[1])[:-1] for index, sat in
         enumerate(sp3_initial_sat)])

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
    list_of_positions = []
    new_system = np.array([coords(*sat)[:-1] for sat in sp3_initial_sat])

    for i in range(16):
        new_system_with_error = np.array([coords(sat[0] + skekkja, sat[1])[:-1] if i & (1 << index) else coords(sat[0] - skekkja, sat[1])[:-1] for index, sat in enumerate(sp3_initial_sat)])
        # setja réttan tíma á skekkjukerfið
        for index, sat_pos in enumerate(new_system_with_error):
            print(new_system_with_error[index][-1])
            print(new_system[-1])
            sat_pos[-1] = new_system[index][-1]

        n4error = Newton(new_system_with_error)
        list_of_positions.append(n4error.GaussNewton(x0, tolerance))
    print(f"max: {max([point_diff(x0, position) for position in list_of_positions]):.7f}")
    print(f"min: {min([point_diff(x0, position) for position in list_of_positions]):.7f}")

def spurning5(plot = True):

    print("---- svar 5 ----- :")

    sp5_initial_sat = np.array([[np.pi / 2, np.pi / 2],  # φ, θ, phi, theta
                            [np.pi / 2, np.pi / 2],
                            [np.pi / 2, np.pi / 2],
                            [np.pi / 2, np.pi / 2],])

    '''
    # búa til staðsetningu frá akkúrat sama stað, og breyta henni smá
    skekkja5 = 1
    #skekkja5 er scali fyrir breytinguna
    for i in range(4):
        sp5_initial_sat[i][0] += (random.random()-.5) * skekkja5
        sp5_initial_sat[i][1] += (random.random()-.5) * skekkja5
    print(sp5_initial_sat)

    '''

    '''
    # staðsetning frá akkúrat sama stað, hliðrað um skekkja5 = 1
    sp5_initial_sat = np.array([[1.52934999, 1.77616402],
                                [1.64586837, 1.54972977],
                                [1.23058977, 1.25151246],
                                [1.76452598, 1.96492466], ])
    

    '''
    # staðsetning frá akkúrat sama stað, hliðrað um skekkja5 = 0.1
    sp5_initial_sat = np.array([ [1.55285912, 1.599031  ],
                                 [1.53712495, 1.62040946],
                                 [1.57151953, 1.61481681],
                                 [1.56491249, 1.53779567],])
    '''

    # staðsetning frá akkúrat sama stað, hliðrað um skekkja5 = 0.01
    sp5_initial_sat = np.array([ [1.57098865 ,1.57282701],
                                 [1.57508225 ,1.57233899],
                                 [1.5707446  ,1.56733488],
                                 [1.56586073 ,1.56823521],])
    '''


    n5system = [coords(phi, theta)[:-1] for phi, theta in sp5_initial_sat]

    for x in sp5_initial_sat:
        x[0] = x[0] + random.randrange(-1,2,2)*skekkja

    n5systemsat_med_skekkju = sp5_initial_sat

    n5system_skekkju = [coords(phi, theta)[:-1] for phi, theta in n5systemsat_med_skekkju]

    # set inn tímanna án skekkjunnar
    for index, sat_pos in enumerate(n5system):
        n5system_skekkju[index][-1] = sat_pos[-1]


    n5 = Newton(n5system_skekkju)
    if plot:
        plot3d(n5.system)
    print(n5.GaussNewton(x0, tolerance))
    print(f"Skekkja: {point_diff(x0,n5.GaussNewton(x0, tolerance)):.7f}")

random_sat_positions = np.array([[nyttSatPos(1) for _ in range(satkerfi_fjoldi)] for _ in range(sample_fjoldi)])

def spurning6(plot=True, calculate_sats=4, skekkja=1e-8):
    if plot:
        print("---- svar 6 ----- :")
    skekkjusafn = []
    for oft in range(0,sample_fjoldi):
        #if oft %100 == 0:
        #    print(oft)

        new_sat_pos = random_sat_positions[oft][:calculate_sats]

        new_system = np.array([coords(*sat)[:-1] for sat in new_sat_pos])
        for i in range(16):
            new_system_with_error = np.empty((0,4))
            for index, sat in enumerate(new_sat_pos):
                if i & (1 << index):
                    new_phi = sat[0] + skekkja
                else:
                    new_phi = sat[0] - skekkja
                new_system_with_error = np.append(new_system_with_error, [coords(new_phi, sat[1])[:-1]], axis=0)
            for index, sat_pos in enumerate(new_system):
                new_system_with_error[index][-1] = sat_pos[-1]
            n6 = Newton(new_system_with_error)
            mismunur = point_diff(x0, n6.GaussNewton(x0, tolerance))*1000

            #skoða skekkju outliers
            if mismunur > 0.005 * 2*10 and plot and False:
                print(str(mismunur) + " er mismunurinn á skekkju númer - >" +str(i))
                plot3d(new_system_with_error)

            skekkjusafn.append(mismunur)

    '''
    print("Gervihnettirnir eru " + str(satkerfi_fjoldi) + " talsins")
    print("meðalskekkjan er " + str(statistics.mean(skekkjusafn)))
    print("min er " + str(min(skekkjusafn)))
    print("max er " + str(max(skekkjusafn)))
    print("milligildi er " + str(statistics.median(skekkjusafn)))
    print("staðalfrávik er " + str(statistics.stdev(skekkjusafn)))
    '''

    if plot:
        plt.hist(skekkjusafn, bins=20, edgecolor='black')
        plt.show()
    return skekkjusafn

def spurning7(plot=True):
    print("---- svar 7 ----- :")
    def bisection(f,a,b,tol, hlidrun=0.1):
        if (np.max(f(skekkja=a))-hlidrun)*(np.max(f(skekkja=b))-hlidrun) >= 0:
            print("Bisection method fails.")
            return None
        else:
            fa = np.max(f(skekkja=a))-hlidrun
            while (b-a)/2 > tol:
                c = (a+b)/2
                fc = np.max(f(skekkja=c))-hlidrun
                if fc == 0:
                    break
                if fc * fa<0:
                    b = c
                else:
                    a = c
                    fa = fc
        return (a+b)/2
    b = 1e-8
    a = 1e-12
    tol = 0.1  # [m]
    ideal_skekkja = (bisection(spurning6, a, b, 1e-12))
    print(ideal_skekkja)
    print(np.max(spurning6(skekkja=ideal_skekkja)))
    ii = []
    vals = []
    exit()
    for i in range(0, 100, 10):
        i = 1e-12 * i
        ii.append(i)
        vals.append(np.max(spurning6(plot=False, skekkja=i)))
    print(ii)
    print(vals)
    plt.scatter(ii, vals)
    plt.show()





def spurning8(plot=True):
    if plot:
        print("---- svar 8 ----- :")
    start_tungl = 4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    skekkjusafn = []
    for i in range(start_tungl, start_tungl+2):
        skekkjusafn.append(spurning6(plot=False, calculate_sats=i))
    ax.boxplot(skekkjusafn, positions=[i for i in range(start_tungl, start_tungl + 2)])
    ax.set_xlabel("Fjöldi tungla")
    ax.set_ylabel("skekkja[m]")
    plt.show()
def spurning9(plot=True):
    print("---- svar 9 ----- :")

    start_tungl = 6
    fig = plt.figure()
    ax = fig.add_subplot(111)
    skekkjusafn = []
    for i in range(start_tungl,satkerfi_fjoldi+1, 1):
        skekkjusafn.append(spurning6(plot=False, calculate_sats=i))
    ax.boxplot(skekkjusafn, positions=[i for i in range(start_tungl, satkerfi_fjoldi+1)])
    ax.set_xlabel("Fjöldi tungla")
    ax.set_ylabel("skekkja[m]")
    plt.show()


def spurning10():
    skekkja = 1e-7
    def coords10(phi, theta, position, altitude=constaltitude):
        A = altitude * np.sin(phi) * np.cos(theta)
        B = altitude * np.sin(phi) * np.sin(theta)
        C = altitude * np.cos(phi)
        # distance = numpy.sqrt(numpy.power((A-0),2)+numpy.power((B-0),2)+numpy.power((C-6370),2))
        distance = np.sqrt((A - position[0]) ** 2 + (B - position[1]) ** 2 + (C - position[2]) ** 2)
        time = distance / c

        return [A, B, C, time]
    def get_position_abc(index):
        return [0, np.sin(np.pi/180*index)*earthaltitude, np.cos(np.pi/180*index)*earthaltitude, 0]
    def new_system_with_skekkja(index: int, position, initial_sat_pos=sp3_initial_sat, skekkja=skekkja):
        new_sat_pos = initial_sat_pos

        # tungl á að ferðast 30° á unit, er rétt að bæta við 15° og 15° við hvort?
        new_sat_pos = [[sat[0]+ np.pi/12*index, sat[1] + np.pi/12*index] for sat in new_sat_pos]

        new_system = np.array([coords10(*sat, position) for sat in new_sat_pos])

        new_system_with_error = np.array([coords10(sat[0] + skekkja, sat[1], position) if 12 & (1 << index) else coords10(sat[0] - skekkja, sat[1], position) for index, sat in enumerate(new_sat_pos)])
        for index, sat_pos in enumerate(new_system):
            new_system_with_error[index][-1] = sat_pos[-1]
        return new_system_with_error
        # tekur inn index og skilar nýrri staðsetningu á gervitunglum með fastri skekkju !!! kannski breytilegri skekkju síðar


    # þetta á að simulera ferðalag frá norðurpól á miðbaug
    # einhversstaðar á leiðinni á eitt tungl að bila, annaðhvort tíminn að byrja að drifta eða hoppa í tíma
    # eftir einhvern tíma þá á bilaða tunglið að verða tekið út og hætta að senda merki
    # plotta upp meðalskekkju á leiðinni, etv. plotta staðsetningu á hnetti með kúlum þar sem stærð kúlu sýnir mestu eða meðalskekkju
    # gerum ráð fyrir að við ferðumst 10000km á 90klst ~= 111km/h,  eitt hopp sé 1 klst
    # eitt tungl ferðast 14000km/h, sem er ca 30°
    # gerum ráð fyrir að merkið berist í gegnum jörðina til að byrja með
    for i in range(0, 90):
        earth_location = np.array(get_position_abc(i))
        new_sys = new_system_with_skekkja(i, earth_location, skekkja=skekkja)
        n10 = Newton(new_sys)
        print(point_diff(n10.GaussNewton(earth_location, 0.1), earth_location)*1000)
        # todo: búa til 3d plot animation
        #plot3d(new_sys)




if __name__ == '__main__':
    #spurning1()
    #spurning2()
    #spurning3()
    #spurning4()
    #spurning5()
    #spurning6(plot=False)
    #spurning7()
    #spurning8()
    #spurning9()
    spurning10()

    #plot3d(new_system)
