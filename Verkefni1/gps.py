import numpy as np
import numpy.linalg as la


if __name__ == '__main__':
    # fylki:
    # 1 2
    # 3 4
    a = np.array([[1, 2], [3, 4]])
    # vigur [1 2]
    b = np.array([1,2])

    print(la.solve(a,b))
