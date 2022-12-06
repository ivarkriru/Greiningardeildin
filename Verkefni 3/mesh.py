import numpy as np
import matplotlib.pyplot as plt
from F import F, F_test, F_str
L = 1
delta = 0.1
H = 5e-3
K_ = 1.68
def mesh(x_min, x_max, y_min, y_max, n,m):
    dx = (x_max - x_min) / n
    dy = (y_max - y_min) / m

    u = np.zeros((n, m))

    # setja upphafsgildi
    for i in range(1, n):
        for j in range(1, m):
            pass
            #
            #u[i][j] = u_0(x[i], y[j])
    midjupunktur = int(n/2)
    kvart = int(n/4)
    for i in range(midjupunktur-kvart,midjupunktur+kvart):
        u[i][0] = 100  # test fyrir upphafsgildi þarf að laga

    # Iterate over the grid points and update the values of u using the finite difference formula
    for i in range(1, n-1):
        for j in range(1, m-1):
            #u[i][j] = (2*u[i-1][j] - 2*u[i+1][j] + u[i][j+1] + 2*u[i][j-1]) / 4
            u[i][j] = (u[i+1][j] + u[i-1][j]) / 2 + (u[i][j+1] + u[i][j-1]) / 2 + 2*H/K/d*u[i][j]
    return u
def u(x, y):
    return 1

def pde(x_min, x_max, y_min, y_max, n,m, Lp, P):
    debug = True
    f = F_test(P, L, delta, K_)
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
        v[j][i] = P
    if debug: print("Power", v[::-1])

    ###################################
    ######### setja inn innri #########
    ###################################

    for i in range(1, m-1):
        for j in range(1, n-1):
            v[j][i] = f.innri(i, j, h)
    if debug: print("innri", v[::-1])
    if debug: print(v.ravel())

    # ítra í gegnum A
    for i in range(0, m*n):
        # pseudocode
        def wij(i_, diff):
            u = [-4, 1, 0,  1, 1, 0, 1]
            return u[diff]
        try:
            A[i][i] = wij(i, 0)
            A[i][i-1] = wij(i, -1)
            A[i][i-4] = wij(i, -4)
            A[i][i+1] = wij(i, +1)
            A[i][i+4] = wij(i, +4)
        except IndexError:  # við vorum að reyna að gera eitthvað útfyrir fylki
            print(i)



    midjupunktur = int(n/2)
    kvart = int(n/4)
    for i in range(midjupunktur-kvart,midjupunktur+kvart):
        u[i][0] = 100  # test fyrir upphafsgildi þarf að laga

    for i in range(1, n-1):
        for j in range(1, m-1):
            #u[i][j] = (2*u[i-1][j] - 2*u[i+1][j] + u[i][j+1] + 2*u[i][j-1]) / 4
            u[i][j] = (u[i+1][j] + u[i-1][j]) / 2 + (u[i][j+1] + u[i][j-1]) / 2 + 2*H/K/d*u[i][j]
    #ax = np.asarray(ax).ravel() # til að breyta array í vigur
    return u

# def poisson(xl, xr, yb, yt, M, N):
#     f = lambda x, y: 0 # define input function data
#     g1 = lambda x: log(x**2 + 1) # define boundary values
#     g2 = lambda x: log(x**2 + 4) # Example 8.8 is shown
#     g3 = lambda y: 2*log(y)
#     g4 = lambda y: log(y**2 + 1)
#     m = M + 1
#     n = N + 1
#     mn = m*n
#     h = (xr-xl)/M
#     h2 = h**2
#     k = (yt-yb)/N
#     k2 = k**2
#     x = xl + (0:M)*h # set mesh values
#     y = yb + (0:N)*k
#     A = zeros(mn, mn)
#     b = zeros(mn, 1)
#     for i in range(2, m-1): # interior points
#         for j in range(2, n-1):
#             A[i + (j-1)*m, i-1 + (j-1)*m] = 1/h2
#             A[i + (j-1)*m, i+1 + (j-1)*m] = 1/h2
#             A[i + (j-1)*m, i + (j-1)*m] = -2/h2 - 2/k2
#             A[i + (j-1)*m, i + (j-2)*m] = 1/k2
#             A[i + (j-1)*m, i + j*m] = 1/k2
#             b[i + (j-1)*m] = f(x[i], y[j])
#     for i in range(0, m): # bottom and top boundary points
#         j = 1
#         A[i + (j-1)*m, i + (j-1)*m] = 1
#         b[i + (j-1)*m] = g1(x[i])
#         j = n
#         A[i + (j-1)*m, i + (j-1)*m] = 1
#         b[i + (j-1)*m] = g2(x[i])
#     for j in range(2, n-1): # left and right boundary points
#         i = 1
#         A[i + (j-1)*m, i + (j-1)*m] = 1
#         b[i + (j-1)*m] = g3(y[j])

if __name__ == '__main__':
    n, m = 10, 10
    Lx, Ly = 2, 2
    Lp = 2
    # Lp = (0,2)
    P = 1
    # ef Lp er tuple, (0,1) þá er [0] min gildið og [1] er max gildið,
    # ef Lp er float þá er powerið miðjað á gridið að lengd Lp
    pde(0, Lx, 0, Ly, n, m, Lp, P)
    meshh = mesh(0, 5, 0, 5, 20, 20)
    for mes in meshh:
        for num in mes:
            print(f"{num:.00f}", end="\t")
        print()

    plt.pcolormesh(meshh)
    plt.show()
