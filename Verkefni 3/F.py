def u(i, j):
    return i*j
class F:
    def __init__(self, P, L, delta, K, H):
        self.P = P
        self.L = L
        self.delta = delta
        self.K = K
        self.H = H

    def nidri(self, i, j, h):
        k1 = 3*u(i,j)/2/h
        k2 = self.H/self.K*u(i,j)
        k3 = -4 * u(i+1, j) / 2/h
        k4 = u(i+2, j) / 2 / h
        return k1 + k2 + k3 + k4
    def uppi(self, i, j, h):
        k1 = -3*u(i,j)/2/h
        k2 = -self.H/self.K*u(i,j)
        k3 = -4 * u(i-1, j) / 2/h
        k4 = u(i-2, j) / 2 / h
        return k1 + k2 + k3 + k4

    def vinstri(self, i, j, h):
        k1 = -3*u(i,j)
        k2 = 4*u(i, j+h)
        k3 = -u(i, j+h+h)
        k4 = 2*h
        return (k1 + k2 + k3) / k4

    def haegri(self, i, j, h):
        k1 = -3*u(i, j)
        k2 = 4*u(i+h, j)
        k3 = -u(i+2*h, j)
        k4 = -2*h
        return (k1 + k2 + k3) / k4

    def innrii(self, i, j, h):
        k1 = u(i-h, j)
        k2 = -2*u(i,j)
        k3 = u(i+h, j)
        k4 = h*h
        return (k1 + k2 + k3) / k4

    def innrij(self, i, j, h):
        k1 = u(i, j-h)
        k2 = -2*u(i,j)
        k3 = u(i+h, j+h)
        k4 = h*h
        return (k1 + k2 + k3) / k4

    def innri(self, i, j, h):
        return self.innrii(i, j, h) + self.innrij(i, j, h)

    def input(self):
        return self.P/(self.L*self.delta*self.K)
class F_test:
    def __init__(self, P, L, delta, K):
        self.P = P
        self.L = L
        self.delta = delta
        self.K = K
        self.count = 0
    def nidri(self, i, j, h):
        self.count += 1
        return self.count

    def uppi(self, i, j, h):
        self.count += 1
        return self.count

    def vinstri(self, i, j, h):
        self.count += 1
        return self.count
    def haegri(self, i, j, h):
        self.count += 1
        return self.count

    def innrii(self, i, j, h):
        self.count += 1
        return self.count

    def innrij(self, i, j, h):
        self.count += 1
        return self.count

    def innri(self, i, j, h):
        return self.innrii(i, j, h) + self.innrij(i, j, h)

    def input(self):
        return self.P/(self.L*self.delta*self.K)
class F_str:
    def __init__(self, P, L, delta, K):
        self.P = P
        self.L = L
        self.delta = delta
        self.K = K
    @staticmethod
    def nidri(i, j, h):
        k1 = -3*u(i, j)
        k2 = 4*u(i+h, j)
        k3 = -u(i+2*h, j)
        k4 = 2*h
        return  "(-3u(i,j) + 4u(i+h, j) - u(i+2h, j)) / 2h"
        #return (k1 + k2 + k3) / k4

    @staticmethod
    def uppi(i, j, h):
        k1 = -3*u(i, j)
        k2 = 4*u(i+h, j)
        k3 = -u(i+2*h, j)
        k4 = -2*h
        return  "(-3u(i,j) + 4u(i+h, j) - u(i+2h, j)) / -2h"
        #return (k1 + k2 + k3) / k4

    @staticmethod
    def vinstri(i, j, h):
        k1 = -3*u(i, j)
        k2 = 4*u(i+h, j)
        k3 = -u(i+2*h, j)
        k4 = 2*h
        # return (k1 + k2 + k3) / k4
        return "(-3u(i,j) + 4u(i+h, j) - u(i+2h, j)) / 2h"

    @staticmethod
    def haegri(i, j, h):
        k1 = -3*u(i, j)
        k2 = 4*u(i+h, j)
        k3 = -u(i+2*h, j)
        k4 = -2*h
        # return (k1 + k2 + k3) / k4
        return "(-3u(i,j) + 4u(i+h, j) - u(i+2h, j)) / -2h"

    @staticmethod
    def innrii(i, j, h):
        k1 = u(i-h, j)
        k2 = -2*u(i,j)
        k3 = u(i+h, j)
        k4 = h*h
        # return (k1 + k2 + k3) / k4
        return "(u(i-h, j) - 2u(i, j) + u(i+h, j)) / h^2"

    @staticmethod
    def innrij(i, j, h):
        k1 = u(i, j-h)
        k2 = -2*u(i,j)
        k3 = u(i+h, j+h)
        k4 = h*h
        # return (k1 + k2 + k3) / k4
        return "(u(i, j-h) - 2u(i, j) + u(i, j+h)) / h^2"
    def innri(self):
        return self.innrii(0,0,0) + "+" + self.innrij(0,0,0)

    def input(self):
        #return self.P/(self.L*self.delta*self.K)
        return "P/(L*delta*K)"
