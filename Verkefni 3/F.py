def u(x, y):
    return 1
class F:
    def __init__(self, P, L, delta, K):
        self.P = P
        self.L = L
        self.delta = delta
        self.K = K
    @staticmethod
    def nidri(x, y, h):
        k1 = -3*u(x, y)
        k2 = 4*u(x+h, y)
        k3 = -u(x+2*h, y)
        k4 = 2*h
        return (k1 + k2 + k3) / k4

    @staticmethod
    def uppi(x, y, h):
        k1 = -3*u(x, y)
        k2 = 4*u(x+h, y)
        k3 = -u(x+2*h, y)
        k4 = -2*h
        return (k1 + k2 + k3) / k4

    @staticmethod
    def vinstri(x, y, h):
        k1 = -3*u(x, y)
        k2 = 4*u(x+h, y)
        k3 = -u(x+2*h, y)
        k4 = 2*h
        return (k1 + k2 + k3) / k4

    @staticmethod
    def haegri(x, y, h):
        k1 = -3*u(x, y)
        k2 = 4*u(x+h, y)
        k3 = -u(x+2*h, y)
        k4 = -2*h
        return (k1 + k2 + k3) / k4

    @staticmethod
    def innrix(x, y, h):
        k1 = u(x-h, y)
        k2 = -2*u(x,y)
        k3 = u(x+h, y)
        k4 = h*h
        return (k1 + k2 + k3) / k4

    @staticmethod
    def innriy(x, y, h):
        k1 = u(x, y-h)
        k2 = -2*u(x,y)
        k3 = u(x+h, y+h)
        k4 = h*h
        return (k1 + k2 + k3) / k4

    def innri(self, x, y, h):
        return self.innrix(x, y, h) + self.innriy(x, y, h)

    def input(self):
        return self.P/(self.L*self.delta*self.K)
class F_str:
    def __init__(self, P, L, delta, K):
        self.P = P
        self.L = L
        self.delta = delta
        self.K = K
    @staticmethod
    def nidri(x, y, h):
        k1 = -3*u(x, y)
        k2 = 4*u(x+h, y)
        k3 = -u(x+2*h, y)
        k4 = 2*h
        return  "(-3u(x,y) + 4u(x+h, y) - u(x+2h, y)) / 2h"
        #return (k1 + k2 + k3) / k4

    @staticmethod
    def uppi(x, y, h):
        k1 = -3*u(x, y)
        k2 = 4*u(x+h, y)
        k3 = -u(x+2*h, y)
        k4 = -2*h
        return  "(-3u(x,y) + 4u(x+h, y) - u(x+2h, y)) / -2h"
        #return (k1 + k2 + k3) / k4

    @staticmethod
    def vinstri(x, y, h):
        k1 = -3*u(x, y)
        k2 = 4*u(x+h, y)
        k3 = -u(x+2*h, y)
        k4 = 2*h
        # return (k1 + k2 + k3) / k4
        return "(-3u(x,y) + 4u(x+h, y) - u(x+2h, y)) / 2h"

    @staticmethod
    def haegri(x, y, h):
        k1 = -3*u(x, y)
        k2 = 4*u(x+h, y)
        k3 = -u(x+2*h, y)
        k4 = -2*h
        # return (k1 + k2 + k3) / k4
        return "(-3u(x,y) + 4u(x+h, y) - u(x+2h, y)) / -2h"

    @staticmethod
    def innrix(x, y, h):
        k1 = u(x-h, y)
        k2 = -2*u(x,y)
        k3 = u(x+h, y)
        k4 = h*h
        # return (k1 + k2 + k3) / k4
        return "(u(x-h, y) - 2u(x, y) + u(x+h, y)) / h^2"

    @staticmethod
    def innriy(x, y, h):
        k1 = u(x, y-h)
        k2 = -2*u(x,y)
        k3 = u(x+h, y+h)
        k4 = h*h
        # return (k1 + k2 + k3) / k4
        return "(u(x, y-h) - 2u(x, y) + u(x, y+h)) / h^2"
    def innri(self):
        return self.innrix(0,0,0) + "+" + self.innriy(0,0,0)

    def input(self):
        #return self.P/(self.L*self.delta*self.K)
        return "P/(L*delta*K)"
