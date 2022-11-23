import numpy as np
import numpy.linalg as la

class Newton:
    def __init__(self, system):
        self.system = system
        self.c = 299792.458

    def fall(self, x):
        j = 0
        n = []
        for i in self.system:
            if i is not None:
                n.append(self.f(x[0], x[1], x[2], x[3], j))
            j = j + 1
        fall = np.array(n)
        return fall
    def dF(self,x):
        j = 0
        n = []
        for i in self.system:
            if i is not None:
                n.append(self.dFformula(x, j))
            j = j + 1
        fall = np.array(n)
        return fall
    def f(self, x, y, z, d, numOfFunInSys):
        return pow((x - self.system[numOfFunInSys][0]), 2) + pow((y - self.system[numOfFunInSys][1]), 2) + pow((z - self.system[numOfFunInSys][2]),2) - pow(self.c, 2) * pow((self.system[numOfFunInSys][3] - d), 2)

    def dFformula(self, vigur, numOfFunInSys):
        return 2 * vigur[0] - 2 * self.system[numOfFunInSys][0], 2 * vigur[1] - 2 * self.system[numOfFunInSys][1], 2 * vigur[2] - 2 * self.system[numOfFunInSys][2],2 * self.system[numOfFunInSys][3] * pow(self.c, 2) - 2 * pow(self.c, 2) * vigur[3]

    def GaussNewton(self, x0, tol):
        '''x0 er vigur i R^n skilgreindur t.d. sem
        x0=np.array([1,2,3])
        gert ráð fyrir að F(x) og Jacobi fylki DF(x) séu skilgreind annars staðar'''
        x = x0
        oldx = x + 2 * tol
        counter = 0
        AT=np.transpose(self.dF(x0))
        while la.norm(x - oldx, np.inf) > tol:
            oldx = x
            s = -la.solve(np.matmul(AT,self.dF(x0)), np.matmul(AT,self.fall(x)))
            x = x + s
            counter += 1
            if counter >= 15:
                print("------------reiknaði of lengi------------")
                break
        return x
