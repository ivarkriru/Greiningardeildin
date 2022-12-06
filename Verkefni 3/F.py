def u(i, j):
    return i*j
class F:
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
        return (k1 + k2 + k3) / k4

    @staticmethod
    def uppi(i, j, h):
        k1 = -3*u(i, j)
        k2 = 4*u(i+h, j)
        k3 = -u(i+2*h, j)
        k4 = -2*h
        return (k1 + k2 + k3) / k4

    @staticmethod
    def vinstri(i, j, h):
        k1 = -3*u(i, j)
        k2 = 4*u(i+h, j)
        k3 = -u(i+2*h, j)
        k4 = 2*h
        return (k1 + k2 + k3) / k4

    @staticmethod
    def haegri(i, j, h):
        k1 = -3*u(i, j)
        k2 = 4*u(i+h, j)
        k3 = -u(i+2*h, j)
        k4 = -2*h
        return (k1 + k2 + k3) / k4

    @staticmethod
    def innrii(i, j, h):
        k1 = u(i-h, j)
        k2 = -2*u(i,j)
        k3 = u(i+h, j)
        k4 = h*h
        return (k1 + k2 + k3) / k4

    @staticmethod
    def innrij(i, j, h):
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
    @staticmethod
    def nidri(i, j, h):
        return i+j

    @staticmethod
    def uppi(i, j, h):
        return i+j

    @staticmethod
    def vinstri(i, j, h):
        return i+j
    @staticmethod
    def haegri(i, j, h):
        return i+j

    @staticmethod
    def innrii(i, j, h):
        return i+j

    @staticmethod
    def innrij(i, j, h):
        return i+j

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
