import time

import numpy as np
import matplotlib.pyplot as plt
from F import F, F_test, F_str


def bua_til_fylki(x_min, x_max, y_min, y_max, n, m, Lp, P, H, K, delta, L_):
    # K = 1.68
    # H = 0.005
    #
    # P = 5
    h = (x_max - x_min) / (m-1)
    k = (y_max - y_min) / (n-1)
    #print(h, k)
    # delta = 0.1
    A = np.zeros((m * n, m * n))
    b = np.zeros((m * n, 1))

    if type(Lp) is tuple:
        if (Lp[1] - Lp[0]) > y_max:
            raise ValueError("Lp má ekki vera stærra en y_max")

        Lmin = int(Lp[0]/k)
        Lmax = int(np.floor((Lp[1]/k)))
    else:
        if Lp > y_max:
            raise ValueError("Lp má ekki vera stærra en y_max")
        # hérna er best ef h gengur upp í Lp en þarf að testa
        padding = n - np.round(Lp/k)  # n[i] - Lp[cm]/h[cm/i]
        padding /= 2
        Lmin = int(padding)
        Lmax = int(n - padding)
    # innra

    for i in range(1, m-1):
        for j in range(1, n-1):
            t = i + (j) * (m)
            A[t][t] = -2 / h ** 2 - 2 / k ** 2 - 2 * H / (K * delta)
            A[t][t + 1] = 1 / h ** 2
            A[t][t - 1] = 1 / h ** 2
            A[t][t + m] = 1 / k ** 2
            A[t][t - m] = 1 / k ** 2

    # vinstri POWER
    for j in range(Lmin, Lmax):
        i = 0
        t = i + (j) * (m)
        A[t][t] = -3 / (2 * h)
        A[t][t + 1] = 2 / h
        A[t][t + 2] = -1 / (2 * h)

    # vinstri no POWER
    for j in range(0, Lmin):
        i = 0
        t = i + (j) * (m)
        A[t][t] = -3 / (2 * h) + H/K
        A[t][t + 1] = 2 / h
        A[t][t + 2] = -1 / (2 * h)
    for j in range(Lmax, n):
        i = 0
        t = i + (j) * (m)
        A[t][t] = -3 / (2 * h) + H/K
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
    for j in range(Lmin, Lmax+1):
        i = 0
        t = i + (j) * (m)
        b[t] = -P / (L_ * delta * K)

    return A, b

def spurning4():
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

if __name__ == '__main__':
    n, m = 101, 11
    Lx, Ly = 4, 4
    Lp = (1,4)
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

    #print("fylki:")
    #print(A)
    # for i in range(m*m):
    #     for j in range(n*n):
    #         print(f"{A[i,j]:.02f}", end="\t")
    #     print()
    #print(b)
    v = np.linalg.solve(A, b) + umhverfishiti
    print("Fyrir "+ str(n) +" X "+ str(m)+ " fylki er tíminn: "  f"{time.time()-t0:.02f}s")
    w = v.reshape((n, m))
    print("Hitastig áætlað: "+ str(v[0,0]))
    print()
    #print(b)
    #print("bla")
    #print(v)
    #print(v.reshape((n,m)))

    #print("max: ", np.max(w))
    # for i in range(m):
    #     for j in range(n):
    #         print(f"{w[i,j]:.02f}", end="\t")
    #     print()
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    from mpl_toolkits.mplot3d import Axes3D
    X, Y = np.meshgrid([i for i in range(m)], [i for i in range(n)])  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, w)
    plt.figure()
    plt.pcolormesh(w)
    plt.show()
