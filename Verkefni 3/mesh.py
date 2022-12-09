import operator
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
np.set_printoptions(linewidth=500)
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
        Lengd_power_min = int(np.ceil(Lengd_power[0] / k_yskref))
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
    #print("Power min : " + str(Lengd_power_min), end=" ")
    #print("Power max : " + str(Lengd_power_max))
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
    # print("power:", Lengd_power_min, Lengd_power_max+1)
    for j in range(Lengd_power_min, Lengd_power_max+1):
        i = 0
        t = i + (j) * (mesh_m)
        A_fylki[t][t] = -3 / (2 * h_xskref)
        A_fylki[t][t + 1] = 2 / h_xskref
        A_fylki[t][t + 2] = -1 / (2 * h_xskref)

    # vinstri no POWER
    # print("ekki power lower:", 0, Lengd_power_min)
    for j in range(0, Lengd_power_min):
        i = 0
        t = i + (j) * (mesh_m)
        A_fylki[t][t] = -3 / (2 * h_xskref) + Heattransfer_co / Kthermal_cond
        A_fylki[t][t + 1] = 2 / h_xskref
        A_fylki[t][t + 2] = -1 / (2 * h_xskref)
    # print("ekki power upper:", Lengd_power_max+1, mesh_n)
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
        if False:
            x = (j/mesh_n)
            Power = (np.sinc((j-25)/2.5))*150
            print(j, x, Power)
        b_fylki[t] = -Power / (lengd_orgjorva * delta * Kthermal_cond)

    return A_fylki, b_fylki

def plotlausn3d(w, xlabel="X", ylabel="Y", zlabel="Z", titill="",log=False,colorbartitill = "Celsius°",xticks="",yticks=""):
    hf = plt.figure()
    ax = plt.axes(projection='3d')

    # Create the contour plot
    X = [*range( 0,w.shape[0])]
    Y = [*range(0,w.shape[1])]
    ax.contourf3D(X, Y, w,200, cmap="viridis")

    # Create a ScalarMappable and set the color limits
    sm = cm.ScalarMappable(cmap="viridis")
    sm.set_array(w)
    sm.set_clim(np.min(w), np.max(w))

    # Add the colorbar
    cb = hf.colorbar(sm, ax=ax, shrink=0.7, pad=0.15)
    cb.set_label(colorbartitill)
    # Add a title
    if xticks != "":
        plt.xticks([0,10,20,30,40],xticks)
    if yticks != "":
        plt.yticks([0, 10, 20, 30, 40], yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if log:
        ax.set_zlabel("log af " + zlabel)

    plt.title(r""+str(titill))
    # Show the plot
    plt.show()



def spurning1():
    '''

    Er í skýrslunni


    '''
    pass

def spurning2():
    '''

    Er í skýrslunni

    '''
    pass

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
    print("Niðurstöður fyrir svar við lið 3")
    print(str(mesh_i_n) + " X " + str(mesh_j_m) + " fylki er "  f"{time.time() - t0:.02f}s"+ " að keyra.")
    w = v.reshape((mesh_i_n, mesh_j_m))
    print(f"Hitastig í (0,0): {w[0,0]:.04f}")
    print(f"Hitastig í (0,Ly): {w[-1,0]:.04f}")
    plotlausn3d(w, xlabel="n", ylabel="m", zlabel="Celsius°", titill="Sp.3 - Hitastig á stöku blaði, n=m=10",log=False,colorbartitill = "Celsius°")


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
    fylki9X9=np.zeros( (9, 9) )
    timafylki9X9=np.zeros( (9, 9) )
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
                nnew=int((n/10)-1)
                mnew=int((m/10)-1)


                arr[count] = {"n": n, "m": m, "temp": temp, "reasonable_estimate": n100m100_hiti, "timi": time.time()-t0}
                count +=1
        print("Reikna 81 kerfi f. sp 4: "  f"{time.time() - t_total:.02f}s")
        print(arr[0])
        #print(arr)
        np.savez('sp4.npz', arr=arr)
    else:
        arr = np.load('sp4.npz', allow_pickle=True)['arr']

    for result in arr:
        fylki9X9[int(result['m']/10)-1][int(result['n']/10)-1] = abs(result['temp']- n100m100_hiti)
        timafylki9X9[int(result['m']/10)-1][int(result['n']/10)-1] = result['timi']
    new_results = []
    for result in arr:
        diff = abs(result['reasonable_estimate'] - result['temp'])
        result['diff'] = diff
        timi = result['timi']
        if diff < 0.01 and timi < 0.5:
            result['diff'] = diff
            #print(result, diff)

            new_results.append(result)
    print("Allar niðurstöður ")
    np.set_printoptions(suppress=True, precision=5)
    print('Fylki 9X9:')
    for i in fylki9X9[::-1]:
        print(i)
    print("timafylki")
    for i in timafylki9X9[::-1]:
        print(i)
    print()
    print('Niðurstöður raðað upp með upplýsingum:')
    for result in arr:
        print(result)
    print()
    new_results.sort(key=operator.itemgetter('diff'))

    print("Úrtak til skoðunar til að sjá breytingu á n og m:")
    i=0
    listimednm=[4,36,40,44,76]
    for nextresult in arr:
        if i in listimednm:
            print(nextresult)
        i=i+1
    print()
    print("Gildi sem eru <0.01 í mun og <0.5 raðað í tímaröð:")
    for result in new_results:
        print(result)
    print()

    diff_array = [[0 for _ in range(9)] for _ in range(9)]
    for result in arr:
        n = int(result['n']/10-1)
        m = int(result['m']/10-1)
        diff_array[n][m] = result['diff']
#    for row in diff_array:
#        print(row)
    timi_array = [[0 for _ in range(9)] for _ in range(9)]
    for result in arr:
        n = int(result['n']/10-1)
        m = int(result['m']/10-1)
        timi_array[n][m] = result['timi']
#    for row in timi_array:
#        print(row)
    diff_array = np.array(diff_array)
    timi_array = np.array(timi_array)
    #timi_array = np.log(timi_array)
    # todo: búa til log 3d plot, þarf bara að taka log af diff_array, timi_array, laga zticks á z ás til að það sé log
    #plotlausn3d(diff_array)
    #plotlausn3d(timi_array,colorbartitill="Tími (s)",log=True)

def spurning5():

    # stærð á meshinu sem reiknar út hitadreyfinguna
    mesh_i_n = 40
    mesh_j_m = 40

    lengdfrax = 0
    lengdfray = 0
    lengdtilx = 4
    lengdtily = 4

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
    print("Niðurstöður fyrir svar við lið 5")
    print("Notað var " + str(mesh_i_n) + " X " + str(mesh_j_m) + " fylki sem var "  f"{time.time() - t0:.02f}s"+ " að keyra.")
    w = v.reshape((mesh_i_n, mesh_j_m))
    print(f"Hæsta hitastig: {np.max(w):.04f}")
    print(f"Hitastig í (0,0): {w[0,0]:.04f}")
    print(f"Hitastig í (0,Ly): {w[-1,0]:.04f}")
    plotlausn3d(w=w, xlabel="cm", ylabel="cm", titill="Dreifing hita í blaði, Lx=4cm Ly=4cm, L=2", xticks=[0, 0.5, 1, 1.5, 2], yticks=[0, 0.5, 1, 1.5, 2])


def spurning6():
    n, m = 40, 40   # 40*40 var valið sem gott compromise milli tíma og skekkju,
                    # þurftum best resolution í báðar áttir því við vorum ekki lengur að horfa á uniform breytingu mv. m ás
    Lx, Ly = 4, 4
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
    skref = int(n/2)
    breyting_per_skref = Ly/n
    arr = [[]]*(skref+1)
    count = 0
    t_total = time.time()
    reikna_upp_a_nytt = False
    if reikna_upp_a_nytt:
        for i in range(0, skref+1):  # +1 til að fá endabilið (2,4)
            t0 = time.time()
            L = 2
            Lp = (0+i*breyting_per_skref, i*breyting_per_skref+L)
            print(f"Lmin: {Lp[0]:.01f}, Lmax: {Lp[1]:.01f}   ", end="")
            A, b= bua_til_fylki(x_min=0, x_max=Lx, y_min=0, y_max=Ly, mesh_n=n,
                                       mesh_m=m, Lengd_power=Lp, Power=P, Heattransfer_co=H,
                                       Kthermal_cond=K, delta=delta)
            max_temp = np.max(np.linalg.solve(A, b)) + umhverfishiti  # beðið var um að taka hámarks hitastig svo max er tekið
            arr[count] = {"lengd_power": Lp,  "timi": time.time()-t0, "max_temp": max_temp}
            count +=1
        print(f"Reikna {count-1} kerfi f. sp 4: "  f"{time.time() - t_total:.02f}s")
        #print(arr[0])
        #print(arr)
        np.savez('sp6.npz', arr=arr)
    else:
        arr = np.load('sp6.npz', allow_pickle=True)['arr']
    if type(arr) is not list:
        arr = arr.tolist()
    arr.sort(key=operator.itemgetter('max_temp'))
    x = []
    y = []
    for row in arr:
        print(row)
        x.append(row['lengd_power'][0])
        y.append(row['max_temp'])
    plt.scatter(x, y)
    plt.xlabel("færsla frá botni[cm]")
    plt.ylabel("hæsti hiti [°C]")
    plt.savefig('6a.png')
    plt.show()

def spurning7():
    from Verkefni1.newton import bisection
    mesh_i_n = 40
    mesh_j_m = 40

    lengdfrax = 0
    lengdfray = 0
    lengdtilx = 4
    lengdtily = 4
    delta = 0.1
    Heattransfer_co = 0.005
    K_thermal_cond = 1.68

    Lengd_power = (0, 2)
    Power = 5
    umhverfishiti = 20
    def bisecfall(Power_):
        Afylki, bfylki = bua_til_fylki(x_min=lengdfrax, x_max=lengdtilx, y_min=lengdfray, y_max=lengdtily, mesh_n=mesh_i_n,
                                       mesh_m=mesh_j_m, Lengd_power=Lengd_power, Power=Power_, Heattransfer_co=Heattransfer_co,
                                       Kthermal_cond=K_thermal_cond, delta=delta)

        v = np.linalg.solve(Afylki, bfylki) + umhverfishiti

        return np.max(v) - 100
    # stærð á meshinu sem reiknar út hitadreyfinguna

    # ef Lp er float þá er powerið miðjað á gridið að lengd Lp

    t0 = time.time()
    a = 1  # Power
    b = 10  # Power
    ideal_skekkja = (bisection(bisecfall, a, b, 1e-15))#f'{num:.3}'
    print(ideal_skekkja)

def spurning8():
    from Verkefni1.newton import bisection
    t0 = time.time()
    print("[",end="")
    svarx = []
    svary = []
    calc_again = False
    if calc_again:
        for K_ in range(1,6):
            #K_ /= 10
            mesh_i_n = 40
            mesh_j_m = 40

            lengdfrax = 0
            lengdfray = 0
            lengdtilx = 4
            lengdtily = 4
            delta = 0.1
            Heattransfer_co = 0.005
            K_thermal_cond = 1.68

            Lengd_power = (0, 2)
            Power = 5
            umhverfishiti = 20
            def bisecfall(Power_):
                Afylki, bfylki = bua_til_fylki(x_min=lengdfrax, x_max=lengdtilx, y_min=lengdfray, y_max=lengdtily, mesh_n=mesh_i_n,
                                               mesh_m=mesh_j_m, Lengd_power=Lengd_power, Power=Power_, Heattransfer_co=Heattransfer_co,
                                               Kthermal_cond=K_, delta=delta)

                v = np.linalg.solve(Afylki, bfylki) + umhverfishiti

                return np.max(v) - 100
            # stærð á meshinu sem reiknar út hitadreyfinguna

            # ef Lp er float þá er powerið miðjað á gridið að lengd Lp

            a = 1  # Power
            b = 30  # Power
            ideal_skekkja = (bisection(bisecfall, a, b, 1e-3))#f'{num:.3}'

            Afylki, bfylki = bua_til_fylki(x_min=lengdfrax, x_max=lengdtilx, y_min=lengdfray, y_max=lengdtily, mesh_n=mesh_i_n,
                                           mesh_m=mesh_j_m, Lengd_power=Lengd_power, Power=ideal_skekkja, Heattransfer_co=Heattransfer_co,
                                           Kthermal_cond=K_, delta=delta)

            v = np.linalg.solve(Afylki, bfylki) + umhverfishiti
            svarx.append(K_)
            svary.append(ideal_skekkja)
            print(ideal_skekkja, K_, np.max(v))
            print(f"[{ideal_skekkja:0.3f},\t {K_:.1f}]",end=",")
    else:
        svar = np.array([[1.642,	 0.1],[2.493,	 0.2],[3.187,	 0.3],[3.780,	 0.4],[4.297,	 0.5],[4.752,	 0.6],[5.157,	 0.7],[5.519,	 0.8],[5.846,	 0.9],[6.143,	 1.0],[6.413,	 1.1],[6.659,	 1.2],[6.886,	 1.3],[7.095,	 1.4],[7.289,	 1.5],[7.468,	 1.6],[7.634,	 1.7],[7.790,	 1.8],[7.935,	 1.9],[8.071,	 2.0],[8.199,	 2.1],[8.319,	 2.2],[8.432,	 2.3],[8.539,	 2.4],[8.640,	 2.5],[8.736,	 2.6],[8.826,	 2.7],[8.912,	 2.8],[8.994,	 2.9],[9.072,	 3.0],[9.146,	 3.1],[9.217,	 3.2],[9.285,	 3.3],[9.350,	 3.4],[9.411,	 3.5],[9.471,	 3.6],[9.528,	 3.7],[9.582,	 3.8],[9.635,	 3.9],[9.685,	 4.0],[9.734,	 4.1],[9.780,	 4.2],[9.825,	 4.3],[9.869,	 4.4],[9.911,	 4.5],[9.951,	 4.6],[9.990,	 4.7],[10.028,	 4.8],[10.064,	 4.9],[10.099,	 5.0],])
    print("]\n")
    print(time.time()-t0)
    if calc_again:
        plt.scatter(svarx, svary)
    else:
        for x in svar:
            plt.scatter(x[1],x[0],c="orange")
    plt.xlabel("Mismunandi gildi á K (W/cm°C) ")
    plt.ylabel("Afl fyrir <100° (W)")

    plt.show()
    #plt.pause(100)

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
