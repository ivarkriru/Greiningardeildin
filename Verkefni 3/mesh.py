import time

import numpy as np
import matplotlib.pyplot as plt
from F import F, F_test, F_str
def pde(x_min, x_max, y_min, y_max, n,m, Lp, P, H):
    debug = True
    f = F_test(P, L, delta, K, H)
    A = np.identity(n*m)
    # b er boundaries
    b = np.zeros((1, m*n))  # ath, kannski þarf að bylta
    v = np.zeros((m, n)) # höfum hann 2d til að einfalda innsetningar, breytum svo í vigur í restina
    h = (x_max - x_min) / m
    k = (y_max - y_min) / n
    ###
    # reikna út power in
    if type(Lp) is tuple:
        if (Lp[1] - Lp[0]) > y_max:
            raise ValueError("Lp má ekki vera stærra en y_max")

        Lmin = int(Lp[0]/h)
        Lmax = int(Lp[1]/h)
    else:
        if Lp > y_max:
            raise ValueError("Lp má ekki vera stærra en y_max")
        # hérna er best ef h gengur upp í Lp en þarf að testa
        padding = n - np.floor(Lp/h)  # n[i] - Lp[cm]/h[cm/i]
        padding /= 2
        Lmin = int(padding)
        Lmax = int(n - padding)

    u = np.zeros((n, m))
    ###################################
    ###### setja inn upphafsgildi #####
    ###################################
    # m hleypur á x og n hleypur á y

    #setja niðri
    for i in range(1, m):
        #v[y][x] = v[j][i]
        j = 0
        v[j][i] = f.nidri(i, j, h)
    if debug: print("niðri", v[::-1])

    # setja hægri
    for j in range(1, n):
        #v[y][x] = v[j][i]
        i=m-1
        v[j][i] = f.haegri(i, j, k)
    if debug: print("haegri", v[::-1])

    # uppi
    for i in range(m-2, 0, -1):  # -1 því við viljum að upphafsgildi fyrir vigurinn endi ekki í horninu
        #v[y][x] = v[j][i]
        j = n-1
        v[j][i] = f.uppi(i, j, h)
    if debug: print("uppi", v[::-1])

    # setja vinstri
    for j in range(n-1, 0, -1):
        #v[y][x] = v[j][i]
        i=0
        v[j][i] = f.vinstri(i, j, k)
    if debug: print("vinstri", v[::-1])

    # setja power inn:
    for j in range(Lmin, Lmax):
        i=0
        v[j][i] = f.input()
    if debug: print("Power", v[::-1])

    ###################################
    ######### setja inn innri #########
    ###################################

    for i in range(1, m-1):
        for j in range(1, n-1):
            v[j][i] = f.innri(i, j, h)
    if debug: print("innri", v[::-1])
    if debug: print(v.ravel())

    # setja inn boundary
    for i in range(0, 5):  # kannski meira eða minna en 5
        def wij(i, j):
            return [-1/2/h, 1, 0, 1, 1, 0, 1]
        try:
            A[i][i] = wij(i, 0)
            if i-1 >= 0:
                A[i][i-1] = wij(i, -1)
            if i-4 >= 0:
                A[i][i-4] = wij(i, -4)
            A[i][i+1] = wij(i, +1)
            A[i][i+4] = wij(i, +4)
        except IndexError:  # við vorum að reyna að gera eitthvað útfyrir fylki
            print(i)

    # ítra í gegnum A
    for i in range(0, m*n):

        def wij(i_, diff):
            u = [-4, 1, 0,  1, 1, 0, 1]
            return u[diff]
        try:
            A[i][i] = wij(i, 0)
            if i-1 >= 0:
                A[i][i-1] = wij(i, -1)
            if i-4 >= 0:
                A[i][i-4] = wij(i, -4)
            A[i][i+1] = wij(i, +1)
            A[i][i+4] = wij(i, +4)
        except IndexError:  # við vorum að reyna að gera eitthvað útfyrir fylki
            print(i)

    for i in range(1, n-1):
        for j in range(1, m-1):
            #u[i][j] = (2*u[i-1][j] - 2*u[i+1][j] + u[i][j+1] + 2*u[i][j-1]) / 4
            u[i][j] = (u[i+1][j] + u[i-1][j]) / 2 + (u[i][j+1] + u[i][j-1]) / 2 + 2*H/K_/delta*u[i][j]
    #ax = np.asarray(ax).ravel() # til að breyta array í vigur
    print(A)
    return A, v.ravel()


def bua_til_fylki(x_min, x_max, y_min, y_max, n, m, Lp, P, H, K, delta):
    # K = 1.68
    # H = 0.005
    #
    # P = 5
    h = (x_max - x_min) / (m-1)
    k = (y_max - y_min) / (n-1)
    print(h, k)
    # delta = 0.1
    A = np.zeros((m * n, m * n))
    b = np.zeros((m * n, 1))

    # innra

    for i in range(1, m-1):
        for j in range(1, n-1):
            t = i + (j) * (m)
            A[t][t] = -2 / h ** 2 - 2 / k ** 2 - 2 * H / (K * delta)
            A[t][t + 1] = 1 / h ** 2
            A[t][t - 1] = 1 / h ** 2
            A[t][t + m] = 1 / k ** 2
            A[t][t - m] = 1 / k ** 2

    # vinstri
    for j in range(0, n):
        i = 0
        t = i + (j) * (m)
        A[t][t] = -3 / (2 * h)
        A[t][t + 1] = 2 / h
        A[t][t + 2] = -1 / (2 * h)
    # hægri
    for j in range(0, n):
        i = m-1
        t = i + (j) * (m)
        A[t][t] = -3 / (2 * h) + H / K
        A[t][t - 1] = 2 / h
        A[t][t - 2] = -1 / (2 * h)

    # bottom
    for i in range(1, m-1):
        j = 0
        t = i + (j) * (m)
        A[t][t] = -3 / (2 * k) + H / K
        A[t][t + m] = 2 / k
        A[t][t + 2 * m] = -1 / (2 * k)
    # top
    for i in range(1, m-1):
        j = n-1
        t = i + (j) * (m)
        A[t][t] = -3 / (2 * k) + H / K
        A[t][t - m] = 2 / k
        A[t][t - 2 * m] = -1 / (2 * k)

    #  POWER
    for j in range(0, n):
        i = 0
        t = i + (j) * (m)
        b[t] = -P / (L * delta * K)

    return A, b



if __name__ == '__main__':
    n, m = 50, 50
    Lx, Ly = 2, 2
    Lp = 2
    L = 1
    delta = 0.1
    H = 5e-3
    K = 1.68
    # Lp = (0,2)
    P = 5
    umhverfishiti = 20
    # ef Lp er tuple, (0,1) þá er [0] min gildið og [1] er max gildið,
    # ef Lp er float þá er powerið miðjað á gridið að lengd Lp
    # A, b = pde(0, Lx, 0, Ly, n, m, Lp, P, H)
    print("wha")
    t0 = time.time()
    A, b = bua_til_fylki(0, Lx, 0, Ly, n, m, Lp, P, H, K, delta)

    print("fylki:")
    # for i in range(m*m):
    #     for j in range(n*n):
    #         print(f"{A[i,j]:.02f}", end="\t")
    #     print()
    #print(b)
    v = np.linalg.solve(A, b) + umhverfishiti
    print(f"{time.time()-t0:.02f}s")
    w = v.reshape((m, n))
    print(b)
    print("bla")
    #print(v)
    #print(v.reshape((n,m)))
    print("max: ", np.max(w))
    # for i in range(m):
    #     for j in range(n):
    #         print(f"{w[i,j]:.02f}", end="\t")
    #     print()
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    from mpl_toolkits.mplot3d import Axes3D
    X, Y = np.meshgrid([i for i in range(n)], [i for i in range(m)])  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, w)
    #plt.pcolormesh(w)
    plt.show()
