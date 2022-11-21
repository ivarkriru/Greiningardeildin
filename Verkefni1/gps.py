import numpy as np
import numpy.linalg as la
system = [[5600, 7540, 20140, 0.07074],
 [18760, 2750, 18610, 0.07220],
 [17610, 14630, 13480, 0.07690],
 [19170, 610, 18390, 0.07242]]

vigur = [0, 0, 6370, 0]

if __name__ == '__main__':
    # fylki:
    # 1 2
    # 3 4
    a = np.array([[1, 2], [3, 4]])
    print(a)

    # vigur [1 2]
    b = np.array([1,2])

    print(la.solve(system,vigur))
