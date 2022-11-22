import numpy as np
from numpy import linalg as LA

def newtonmult(x0,tol):
    '''x0 er vigur i R^n skilgreindur t.d. sem
    x0=np.array([1,2,3])
    gert ráð fyrir að F(x) og Jacobi fylki DF(x) séu skilgreind annars staðar'''

    x=x0
    oldx=x+2*tol
    Fx = np.power(x,2)
    print(Fx)
    DFx = 2 * x
    print(DFx)
    while LA.norm(x-oldx,np.inf)>tol:
        oldx=x
        s=-LA.solve(DFx,Fx)
        x=x+s
    return(x)

    
newtonmult(2,1)