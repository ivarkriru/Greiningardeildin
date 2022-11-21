import numpy as np
import numpy.linalg as la
system = np.array([[15600, 7540, 20140, 0.07074],
 [18760, 2750, 18610, 0.07220],
 [17610, 14630, 13480, 0.07690],
 [19170, 610, 18390, 0.07242]])

vigur = [0, 0, 6370, 0]
def vect_jacobian(system):
    jacobi = np.zeros((system.shape[0], (system.shape[1]-1)))
    for i in range(system.shape[1]):
        for j in range(system.shape[0]-1):
            jacobi[i][j] = system[i][j] * (system.shape[0]-1-j)
    return jacobi

def f(vigur):
    return la.solve(vigur, system)

def dF(vigur):
    jacobi = vect_jacobian(system)
    return jacobi
def newton(x0, tolerance):
    oldx = x0 + 2*tolerance
    x = x0
    while abs(oldx-x) > tolerance:
        oldx = x
        x = x - f(x) / dF(x)
    return x



if __name__ == '__main__':
    # fylki:
    # 1 2
    # 3 4
    a = np.array([[1, 2], [3, 4]])
    print(a)

    # vigur [1 2]
    b = np.array([1,2])

    print(la.solve(system,vigur))
    print(vect_jacobian(system))
