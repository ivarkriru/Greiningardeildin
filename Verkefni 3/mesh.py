import operator
import time
import numpy as np
import matplotlib.pyplot as plt

def bua_til_fylki(x_min, x_max, y_min, y_max, mesh_n, mesh_m, Lengd_power, Power, Heattransfer_co, Kthermal_cond, delta):

    h_xskref = (x_max - x_min) / (mesh_m - 1)
    k_yskref = (y_max - y_min) / (mesh_n - 1)

    A_fylki = np.zeros((mesh_m * mesh_n, mesh_m * mesh_n))
    b_fylki = np.zeros((mesh_m * mesh_n, 1))

    # ef Lp er tuple, (0,1) þá er [0] min gildið og [1] er max gildið, t.d.
    if type(Lengd_power) is tuple:
        if (Lengd_power[1] - Lengd_power[0]) > y_max:
            raise ValueError("Lengd aflsins má ekki vera lengri en allur y ásinn")

        lengd_orgjorva = Lengd_power[1] - Lengd_power[0]

        # búum til bil fyrir Lengd power bilið
        Lengd_power_min = int(Lengd_power[0] / k_yskref)
        Lengd_power_max = int(np.floor((Lengd_power[1] / k_yskref)))
    else:
        if Lengd_power > y_max:
            raise ValueError("Lengd aflsins má ekki vera lengri en allur y ásinn")

        lengd_orgjorva = Lengd_power
        # hérna er best ef h gengur upp í Lp en þarf að testa
        padding = mesh_n - np.round(Lengd_power / k_yskref)  # n[i] - Lp[cm]/h[cm/i]
        padding /= 2
        Lengd_power_min = int(padding)
        Lengd_power_max = int(mesh_n - padding)
    print(Lengd_power_min, Lengd_power_max)
    #print(f"aflið er á þessu bili {Lengd_power_min=}, {Lengd_power_max=} ")

    # innra

    for i in range(1, mesh_m - 1):
        for j in range(1, mesh_n - 1):
            # hér stekkur t yfir dálk
            t = i + (j) * (mesh_m)
            A_fylki[t][t] = -2 / h_xskref ** 2 - 2 / k_yskref ** 2 - 2 * Heattransfer_co / (Kthermal_cond * delta)
            # hægri
            A_fylki[t][t + 1] = 1 / h_xskref ** 2
            #vinstri
            A_fylki[t][t - 1] = 1 / h_xskref ** 2
            #neðan
            A_fylki[t][t + mesh_m] = 1 / k_yskref ** 2
            #ofan
            A_fylki[t][t - mesh_m] = 1 / k_yskref ** 2

    # vinstri POWER
    for j in range(Lengd_power_min, Lengd_power_max+1):
        i = 0
        t = i + (j) * (mesh_m)
        A_fylki[t][t] = -3 / (2 * h_xskref)
        A_fylki[t][t + 1] = 2 / h_xskref
        A_fylki[t][t + 2] = -1 / (2 * h_xskref)

    # vinstri no POWER
    for j in range(0, Lengd_power_min):
        i = 0
        t = i + (j) * (mesh_m)
        A_fylki[t][t] = -3 / (2 * h_xskref) + Heattransfer_co / Kthermal_cond
        A_fylki[t][t + 1] = 2 / h_xskref
        A_fylki[t][t + 2] = -1 / (2 * h_xskref)

    for j in range(Lengd_power_max+1, mesh_n):
        i = 0
        t = i + (j) * (mesh_m)
        A_fylki[t][t] = -3 / (2 * h_xskref) + Heattransfer_co / Kthermal_cond
        A_fylki[t][t + 1] = 2 / h_xskref
        A_fylki[t][t + 2] = -1 / (2 * h_xskref)

    # hægri
    for j in range(0, mesh_n):
        i = mesh_m - 1
        t = i + (j) * (mesh_m)
        A_fylki[t][t] = -3 / (2 * h_xskref) + Heattransfer_co / Kthermal_cond
        A_fylki[t][t - 1] = 2 / h_xskref
        A_fylki[t][t - 2] = -1 / (2 * h_xskref)

    # bottom
    for i in range(1, mesh_m - 1):
        j = 0
        t = i + (j) * (mesh_m)
        A_fylki[t][t] = -3 / (2 * k_yskref) + Heattransfer_co / Kthermal_cond
        A_fylki[t][t + mesh_m] = 2 / k_yskref
        A_fylki[t][t + 2 * mesh_m] = -1 / (2 * k_yskref)
    # top
    for i in range(1, mesh_m - 1):
        j = mesh_n - 1
        t = i + (j) * (mesh_m)
        A_fylki[t][t] = -3 / (2 * k_yskref) + Heattransfer_co / Kthermal_cond
        A_fylki[t][t - mesh_m] = 2 / k_yskref
        A_fylki[t][t - 2 * mesh_m] = -1 / (2 * k_yskref)

    #  POWER
    for j in range(Lengd_power_min, Lengd_power_max+1):
        i = 0
        t = i + (j) * (mesh_m)
        b_fylki[t] = -Power / (lengd_orgjorva * delta * Kthermal_cond)

    return A_fylki, b_fylki

def plotlausn3d(w,mesh_i_n,mesh_j_m):
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid([i for i in range(mesh_j_m)],
                       [i for i in range(mesh_i_n)])  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, w)
    plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(r"Titill")
    plt.pcolormesh(w)
    plt.title('matplotlib.pyplot.pcolormesh() function Example', fontweight ="bold")
    plt.show()


def spurning1():
    pass

def spurning2():
    pass

def spurning3rett():
    mesh_i_n = 10
    mesh_j_m = 10
    n, m = 10, 10
    Lx, Ly = 2, 2
    Lp = 2
    L = 2
    delta = 0.1
    H = 0.005
    K = 1.68
    P = 5
    umhverfishiti = 20
    Afylki, bfylki = bua_til_fylki(0, Lx, 0, Ly, n, m, Lp, P, H, K, delta)
    v = np.linalg.solve(Afylki, bfylki) + umhverfishiti
    w = v.reshape((mesh_i_n, mesh_j_m))
    #print(v)
    #print(w)
    print("Hitastig í (0,0): " + str(w[0,0]))
    print("Hitastig í (0,Ly): " + str(w[-1,0]))
    plotlausn3d(w=w, mesh_i_n=mesh_i_n, mesh_j_m=mesh_j_m)

def spurning3():
    # stærð á meshinu sem reiknar út hitadreyfinguna
    mesh_i_n = 10
    mesh_j_m = 10

    lengdfrax = 0
    lengdfray = 0
    lengdtilx = 2
    lengdtily = 2
    delta = 0.1
    Heattransfer_co = 0.005
    K_thermal_cond = 1.68

    Lengd_power = (0, 2)
    Power = 5
    umhverfishiti = 20

    # ef Lp er float þá er powerið miðjað á gridið að lengd Lp


    t0 = time.time()

    Afylki, bfylki = bua_til_fylki(x_min=lengdfrax, x_max=lengdtilx, y_min=lengdfray, y_max=lengdtily, mesh_n=mesh_i_n,
                                   mesh_m=mesh_j_m, Lengd_power=Lengd_power, Power=Power, Heattransfer_co=Heattransfer_co,
                                   Kthermal_cond=K_thermal_cond, delta=delta)

    v = np.linalg.solve(Afylki, bfylki) + umhverfishiti
    print("Fyrir " + str(mesh_i_n) + " X " + str(mesh_j_m) + " fylki er tíminn: "  f"{time.time() - t0:.02f}s")
    w = v.reshape((mesh_i_n, mesh_j_m))

    print("Hitastig áætlað: " + str(v[0, 0]))

    plotlausn3d(w=w,mesh_i_n=mesh_i_n,mesh_j_m=mesh_j_m)


def spurning4():
    n, m = 10, 10
    Lx, Ly = 2, 2
    Lp = 2
    L = 2
    delta = 0.1
    H = 0.005
    K = 1.68
    # Lp = (0,2)
    P = 5
    umhverfishiti = 20
    # ef Lp er tuple, (0,1) þá er [0] min gildið og [1] er max gildið,
    # ef Lp er float þá er powerið miðjað á gridið að lengd Lp
    # A, b = pde(0, Lx, 0, Ly, n, m, Lp, P, H)
    t0 = time.time()
    n100m100_hiti = np.load("n100_m100.npz")['w'][0, 0]
    #arr = np.zeros((9*9, 1))
    arr = [[]]*9*9
    count = 0
    t_total = time.time()
    reikna_upp_a_nytt = False
    if reikna_upp_a_nytt:
        for n in range(10, 100, 10):
            for m in range(10, 100, 10):
                # todo: væri gaman að interpolate'a stóru fylkin í 10x10 og bera saman þannig
                t0 = time.time()
                A, b = bua_til_fylki(0, Lx, 0, Ly, n, m, Lp, P, H, K, delta)
                temp = np.linalg.solve(A, b)[0,0] + umhverfishiti
                arr[count] = {"n": n, "m": m, "temp": temp, "reasonable_estimate": n100m100_hiti, "timi": time.time()-t0}
                count +=1
        print("Reikna 81 kerfi f. sp 4: "  f"{time.time() - t_total:.02f}s")
        print(arr[0])
        #print(arr)
        np.savez('sp4.npz', arr=arr)
    else:
        arr = np.load('sp4.npz', allow_pickle=True)['arr']
        #print(arr)
        new_results = []
        for result in arr:
            diff = abs(result['reasonable_estimate'] - result['temp'])
            timi = result['timi']
            if diff < 0.01 and timi < 0.5:
                result['diff'] = diff
                print(result, diff)

                new_results.append(result)

        new_results.sort(key=operator.itemgetter('diff'))
        print("sorted:")
        for result in new_results:
            print(result)



def spurning5():

    mesh_i_n = 20
    mesh_j_m = 10
    Lx, Ly = 2, 2
    Lp = 2
    L = 2
    delta = 0.1
    H = 0.005
    K = 1.68
    # Lp = (0,2)
    P = 5
    umhverfishiti = 20
    # ef Lp er tuple, (0,1) þá er [0] min gildið og [1] er max gildið,
    # ef Lp er float þá er powerið miðjað á gridið að lengd Lp
    # A, b = pde(0, Lx, 0, Ly, n, m, Lp, P, H)
    t0 = time.time()
    A, b = bua_til_fylki(0, Lx, 0, Ly, mesh_i_n, mesh_j_m, Lp, P, H, K, delta)

def spurning6():
    n, m = 10, 10
    Lx, Ly = 2, 2
    Lp = 2
    delta = 0.1
    H = 0.005
    K = 1.68
    # Lp = (0,2)
    P = 5
    umhverfishiti = 20
    # ef Lp er tuple, (0,1) þá er [0] min gildið og [1] er max gildið,
    # ef Lp er float þá er powerið miðjað á gridið að lengd Lp
    # A, b = pde(0, Lx, 0, Ly, n, m, Lp, P, H)
    t0 = time.time()

    A, b = bua_til_fylki(0, Lx, 0, Ly, n, m, Lp, P, H, K, delta)

def spurning7():
    pass

def spurning8():
    pass

def spurning9():
    pass

def auka():
    pass

'''
from mesh import *

if __name__ == '__main__':
    spurning1()
    spurning2()
    spurning3rett()
    spurning4()
    spurning5()
    spurning6()
    spurning7()
    spurning8()
    spurning9()
    auka()
'''
