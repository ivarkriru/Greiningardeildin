import time
import numpy as np
import matplotlib.pyplot as plt

sp4_arr = [{'n': 10, 'm': 10, 'temp': 164.96257194147285, 'reasonable_estimate': 164.80842887347413}, {'n': 10, 'm': 20, 'temp': 164.85112433118888, 'reasonable_estimate': 164.80842887347413}, {'n': 10, 'm': 30, 'temp': 164.83123765754775, 'reasonable_estimate': 164.80842887347413}, {'n': 10, 'm': 40, 'temp': 164.8243972399306, 'reasonable_estimate': 164.80842887347413}, {'n': 10, 'm': 50, 'temp': 164.82126576777208, 'reasonable_estimate': 164.80842887347413}, {'n': 10, 'm': 60, 'temp': 164.81957857075858, 'reasonable_estimate': 164.80842887347413}, {'n': 10, 'm': 70, 'temp': 164.81856800519918, 'reasonable_estimate': 164.80842887347413}, {'n': 10, 'm': 80, 'temp': 164.8179158815635, 'reasonable_estimate': 164.80842887347413}, {'n': 10, 'm': 90, 'temp': 164.81747110030304, 'reasonable_estimate': 164.80842887347413}, {'n': 20, 'm': 10, 'temp': 164.95622007304385, 'reasonable_estimate': 164.80842887347413}, {'n': 20, 'm': 20, 'temp': 164.8446380818101, 'reasonable_estimate': 164.80842887347413}, {'n': 20, 'm': 30, 'temp': 164.82469746279477, 'reasonable_estimate': 164.80842887347413}, {'n': 20, 'm': 40, 'temp': 164.81782762078166, 'reasonable_estimate': 164.80842887347413}, {'n': 20, 'm': 50, 'temp': 164.81467756209926, 'reasonable_estimate': 164.80842887347413}, {'n': 20, 'm': 60, 'temp': 164.81297754243906, 'reasonable_estimate': 164.80842887347413}, {'n': 20, 'm': 70, 'temp': 164.81195758778338, 'reasonable_estimate': 164.80842887347413}, {'n': 20, 'm': 80, 'temp': 164.8112982913394, 'reasonable_estimate': 164.80842887347413}, {'n': 20, 'm': 90, 'temp': 164.81084785071988, 'reasonable_estimate': 164.80842887347413}, {'n': 30, 'm': 10, 'temp': 164.95507179144474, 'reasonable_estimate': 164.80842887347413}, {'n': 30, 'm': 20, 'temp': 164.84346024285117, 'reasonable_estimate': 164.80842887347413}, {'n': 30, 'm': 30, 'temp': 164.82350666096133, 'reasonable_estimate': 164.80842887347413}, {'n': 30, 'm': 40, 'temp': 164.81662936650463, 'reasonable_estimate': 164.80842887347413}, {'n': 30, 'm': 50, 'temp': 164.8134744369321, 'reasonable_estimate': 164.80842887347413}, {'n': 30, 'm': 60, 'temp': 164.81177097640614, 'reasonable_estimate': 164.80842887347413}, {'n': 30, 'm': 70, 'temp': 164.81074845687317, 'reasonable_estimate': 164.80842887347413}, {'n': 30, 'm': 80, 'temp': 164.81008717326924, 'reasonable_estimate': 164.80842887347413}, {'n': 30, 'm': 90, 'temp': 164.8096351474499, 'reasonable_estimate': 164.80842887347413}, {'n': 40, 'm': 10, 'temp': 164.95467384876505, 'reasonable_estimate': 164.80842887347413}, {'n': 40, 'm': 20, 'temp': 164.843051028189, 'reasonable_estimate': 164.80842887347413}, {'n': 40, 'm': 30, 'temp': 164.82309226558067, 'reasonable_estimate': 164.80842887347413}, {'n': 40, 'm': 40, 'temp': 164.81621189356227, 'reasonable_estimate': 164.80842887347413}, {'n': 40, 'm': 50, 'temp': 164.81305490345395, 'reasonable_estimate': 164.80842887347413}, {'n': 40, 'm': 60, 'temp': 164.8113499609414, 'reasonable_estimate': 164.80842887347413}, {'n': 40, 'm': 70, 'temp': 164.81032632281676, 'reasonable_estimate': 164.80842887347413}, {'n': 40, 'm': 80, 'temp': 164.80966416097615, 'reasonable_estimate': 164.80842887347413}, {'n': 40, 'm': 90, 'temp': 164.80921143082213, 'reasonable_estimate': 164.80842887347413}, {'n': 50, 'm': 10, 'temp': 164.9544906314882, 'reasonable_estimate': 164.80842887347413}, {'n': 50, 'm': 20, 'temp': 164.8428623058373, 'reasonable_estimate': 164.80842887347413}, {'n': 50, 'm': 30, 'temp': 164.82290093662272, 'reasonable_estimate': 164.80842887347413}, {'n': 50, 'm': 40, 'temp': 164.81601898162788, 'reasonable_estimate': 164.80842887347413}, {'n': 50, 'm': 50, 'temp': 164.81286091412989, 'reasonable_estimate': 164.80842887347413}, {'n': 50, 'm': 60, 'temp': 164.81115518535492, 'reasonable_estimate': 164.80842887347413}, {'n': 50, 'm': 70, 'temp': 164.8101309462775, 'reasonable_estimate': 164.80842887347413}, {'n': 50, 'm': 80, 'temp': 164.80946831073385, 'reasonable_estimate': 164.80842887347413}, {'n': 50, 'm': 90, 'temp': 164.80901519421175, 'reasonable_estimate': 164.80842887347413}, {'n': 60, 'm': 10, 'temp': 164.95439143363512, 'reasonable_estimate': 164.80842887347413}, {'n': 60, 'm': 20, 'temp': 164.8427600040171, 'reasonable_estimate': 164.80842887347413}, {'n': 60, 'm': 30, 'temp': 164.82279713349007, 'reasonable_estimate': 164.80842887347413}, {'n': 60, 'm': 40, 'temp': 164.8159142524604, 'reasonable_estimate': 164.80842887347413}, {'n': 60, 'm': 50, 'temp': 164.81275554629357, 'reasonable_estimate': 164.80842887347413}, {'n': 60, 'm': 60, 'temp': 164.81104934487047, 'reasonable_estimate': 164.80842887347413}, {'n': 60, 'm': 70, 'temp': 164.8100247442397, 'reasonable_estimate': 164.80842887347413}, {'n': 60, 'm': 80, 'temp': 164.80936182054018, 'reasonable_estimate': 164.80842887347413}, {'n': 60, 'm': 90, 'temp': 164.8089084674552, 'reasonable_estimate': 164.80842887347413}, {'n': 70, 'm': 10, 'temp': 164.9543317541837, 'reasonable_estimate': 164.80842887347413}, {'n': 70, 'm': 20, 'temp': 164.84269840127354, 'reasonable_estimate': 164.80842887347413}, {'n': 70, 'm': 30, 'temp': 164.8227345845489, 'reasonable_estimate': 164.80842887347413}, {'n': 70, 'm': 40, 'temp': 164.81585111335295, 'reasonable_estimate': 164.80842887347413}, {'n': 70, 'm': 50, 'temp': 164.8126919958437, 'reasonable_estimate': 164.80842887347413}, {'n': 70, 'm': 60, 'temp': 164.81098548879967, 'reasonable_estimate': 164.80842887347413}, {'n': 70, 'm': 70, 'temp': 164.80996064893466, 'reasonable_estimate': 164.80842887347413}, {'n': 70, 'm': 80, 'temp': 164.80929753589845, 'reasonable_estimate': 164.80842887347413}, {'n': 70, 'm': 90, 'temp': 164.80884402581538, 'reasonable_estimate': 164.80842887347413}, {'n': 80, 'm': 10, 'temp': 164.95429308298577, 'reasonable_estimate': 164.80842887347413}, {'n': 80, 'm': 20, 'temp': 164.84265845421967, 'reasonable_estimate': 164.80842887347413}, {'n': 80, 'm': 30, 'temp': 164.82269400145364, 'reasonable_estimate': 164.80842887347413}, {'n': 80, 'm': 40, 'temp': 164.81581013031578, 'reasonable_estimate': 164.80842887347413}, {'n': 80, 'm': 50, 'temp': 164.81265073247044, 'reasonable_estimate': 164.80842887347413}, {'n': 80, 'm': 60, 'temp': 164.8109440145253, 'reasonable_estimate': 164.80842887347413}, {'n': 80, 'm': 70, 'temp': 164.80991901115164, 'reasonable_estimate': 164.80842887347413}, {'n': 80, 'm': 80, 'temp': 164.80925576298657, 'reasonable_estimate': 164.80842887347413}, {'n': 80, 'm': 90, 'temp': 164.80880214458787, 'reasonable_estimate': 164.80842887347413}, {'n': 90, 'm': 10, 'temp': 164.9542666026447, 'reasonable_estimate': 164.80842887347413}, {'n': 90, 'm': 20, 'temp': 164.8426310826472, 'reasonable_estimate': 164.80842887347413}, {'n': 90, 'm': 30, 'temp': 164.82266618366927, 'reasonable_estimate': 164.80842887347413}, {'n': 90, 'm': 40, 'temp': 164.81578202769325, 'reasonable_estimate': 164.80842887347413}, {'n': 90, 'm': 50, 'temp': 164.81262242687527, 'reasonable_estimate': 164.80842887347413}, {'n': 90, 'm': 60, 'temp': 164.81091556118128, 'reasonable_estimate': 164.80842887347413}, {'n': 90, 'm': 70, 'temp': 164.80989043778496, 'reasonable_estimate': 164.80842887347413}, {'n': 90, 'm': 80, 'temp': 164.80922709639725, 'reasonable_estimate': 164.80842887347413}, {'n': 90, 'm': 90, 'temp': 164.80877339598203, 'reasonable_estimate': 164.80842887347413}]
def bua_til_fylki(x_min, x_max, y_min, y_max, mesh_n, mesh_m, Lengd_power, Power, Heattransfer_co, Kthermal_cond, delta, lengd_orgjorva):

    h_xskref = (x_max - x_min) / (mesh_m - 1)
    k_yskref = (y_max - y_min) / (mesh_n - 1)

    A_fylki = np.zeros((mesh_m * mesh_n, mesh_m * mesh_n))
    b_fylki = np.zeros((mesh_m * mesh_n, 1))

    # ef Lp er tuple, (0,1) þá er [0] min gildið og [1] er max gildið, t.d.
    if type(Lengd_power) is tuple:
        if (Lengd_power[1] - Lengd_power[0]) > y_max:
            raise ValueError("Lengd aflsins má ekki vera lengri en allur y ásinn")

        # búum til bil fyrir Lengd power bilið
        Lengd_power_min = int(Lengd_power[0] / k_yskref)
        Lengd_power_max = int(np.floor((Lengd_power[1] / k_yskref)))
    else:
        if Lengd_power > y_max:
            raise ValueError("Lengd aflsins má ekki vera lengri en allur y ásinn")
        # hérna er best ef h gengur upp í Lp en þarf að testa
        padding = mesh_n - np.round(Lengd_power / k_yskref)  # n[i] - Lp[cm]/h[cm/i]
        padding /= 2
        Lengd_power_min = int(padding)
        Lengd_power_max = int(mesh_n - padding)

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
    for j in range(Lengd_power_min, Lengd_power_max):
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

    for j in range(Lengd_power_max, mesh_n):
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
    Afylki, bfylki = bua_til_fylki(0, Lx, 0, Ly, n, m, Lp, P, H, K, delta, L)
    v = np.linalg.solve(Afylki, bfylki) + umhverfishiti
    w = v.reshape((mesh_i_n, mesh_j_m))
    #print(v)
    #print(w)
    print("Hitastig í (0,0): " + str(w[0,0]))
    print("Hitastig í (0,Ly): " + str(w[-1,0]))
    plotlausn3d(w=w, mesh_i_n=mesh_i_n, mesh_j_m=mesh_j_m)

def spurning3():
    # stærð á meshinu sem reiknar út hitadreyfinguna
    mesh_i_n = 80
    mesh_j_m = 80

    lengdfrax = 0
    lengdfray = 0
    lengdtilx = 20
    lengdtily = 10
    Lengd_power = 2
    Lengd_orgjorva = 2
    delta = 0.1
    Heattransfer_co = 0.005
    K_thermal_cond = 1.68

    Lengd_power = (2, 7)
    Power = 5
    umhverfishiti = 20

    # ef Lp er float þá er powerið miðjað á gridið að lengd Lp


    t0 = time.time()

    Afylki, bfylki = bua_til_fylki(x_min=lengdfrax, x_max=lengdtilx, y_min=lengdfray, y_max=lengdtily, mesh_n=mesh_i_n,
                                   mesh_m=mesh_j_m, Lengd_power=Lengd_power, Power=Power, Heattransfer_co=Heattransfer_co,
                                   Kthermal_cond=K_thermal_cond, delta=delta, lengd_orgjorva=Lengd_orgjorva)

    v = np.linalg.solve(Afylki, bfylki) + umhverfishiti
    print("Fyrir " + str(mesh_i_n) + " X " + str(mesh_j_m) + " fylki er tíminn: "  f"{time.time() - t0:.02f}s")
    w = v.reshape((mesh_i_n, mesh_j_m))

    print("Hitastig áætlað: " + str(v[0, 0]))

    plotlausn3d(w=v,mesh_i_n=mesh_i_n,mesh_j_m=mesh_j_m)


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
    for n in range(10, 100, 10):
        for m in range(10, 100, 10):

            A, b = bua_til_fylki(0, Lx, 0, Ly, n, m, Lp, P, H, K, delta, L)
            temp = np.linalg.solve(A, b)[0,0] + umhverfishiti
            arr[count] = {"n": n, "m": m, "temp": temp, "reasonable_estimate": n100m100_hiti}
            count +=1
    print(arr[0])
    #print(arr)
    #np.savez('sp4.npz', arr=arr)
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
    A, b = bua_til_fylki(0, Lx, 0, Ly, mesh_i_n, mesh_j_m, Lp, P, H, K, delta, L)

def spurning6():
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

    A, b = bua_til_fylki(0, Lx, 0, Ly, n, m, Lp, P, H, K, delta, L)

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