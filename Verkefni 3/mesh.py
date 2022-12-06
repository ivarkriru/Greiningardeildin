import numpy as np
import matplotlib.pyplot as plt
from F import F, F_test, F_str
P = 1
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

def pde(x_min, x_max, y_min, y_max, n,m):
    f = F_test(P, L, delta, K_)
    A = np.identity(n*m)
    # b er boundaries
    b = np.zeros((1, m*n))  # ath, kannski þarf að bylta
    v = np.zeros((m, n)) # höfum hann 2d til að einfalda innsetningar, breytum svo í vigur í restina
    h = (x_max - x_min) / n
    k = (y_max - y_min) / m

    u = np.zeros((n, m))

    # m hleypur á x og n hleypur á y
    # setja uppi
    print("uppi")
    for i in range(0, m-1):  # -1 því við viljum að upphafsgildi fyrir vigurinn endi ekki í horninu
        #v[y][x] = v[j][i]
        j = n-1
        v[j][i] = f.uppi(i, j, h)
    print(v)

    print("niðri")
    #setja niðri
    for i in range(0, m-1):
        #v[y][x] = v[j][i]
        j = 0
        v[j][i] = f.nidri(i, j, h)
    print(v)

    print("vinstri")
    # setja vinstri
    for j in range(0, n-1):
        #v[y][x] = v[j][i]
        i=0
        v[j][i] = f.vinstri(i, j, k)
    print(v)
    # setja hægri
    print("haegri")
    for j in range(0, n-1):
        #v[y][x] = v[j][i]
        i=m-1
        v[j][i] = f.haegri(i, j, k)
    print(v)
    print(v.ravel())
    # ítra í gegnum A
    for i in range(1, m*n):
        # pseudocode
        def wij(i_, diff):
            u = [-4, 1, 0,  1, 1, 0, 1]
            return u[diff]
            #if diff == 0:
            #    return u[i][j]
            #if diff == -1:
            #    return u[i][j]
            #return (2*u[i-1][j] + *u[i+1][j] + u[i][j+1] + u[i][j-1]) / 4
        try:
            if not boundary_condition:
                A[i][i] = wij(i, 0)
                A[i][i-1] = wij(i, -1)
                A[i][i-4] = wij(i, -4)
                A[i][i+1] = wij(i, +1)
                A[i][i+4] = wij(i, +4)
            elif i==xmin:
                pass



        except IndexError:  # við vorum að reyna að gera eitthvað útfyrir fylki
            print(i)



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
    pde(0, 5, 0, 5, 6, 6)
    meshh = mesh(0, 5, 0, 5, 20, 20)
    for mes in meshh:
        for num in mes:
            print(f"{num:.00f}", end="\t")
        print()

    plt.pcolormesh(meshh)
    plt.show()
