import random
import time

# import numpy as np
# import matplotlib.pyplot as plt
from adferdir import Foll, Pendulum
import math


def point_diff(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def spurning1(plot=False):
    '''

    er á ipad pdf skjalinu hér á git.

    '''
    pass


def spurning2(plot=False):
    '''

    euler fallið í adferd python skjalinu

    '''
    pass





pi_= {"π/3":math.pi/3, "π/6":math.pi/6, "π/2":math.pi/2, "π":math.pi, "π/4":math.pi/4, 0:0, "π/12":math.pi/12, "-π/12": -math.pi/12}
def spurning9(plot=False):
    follin = Foll()
    p = Pendulum(L_1=22, m_1=11, L_2=5, m_2=107)
    T = 20
    n_to_power_2 = 8  # 0-7
    #pendulalist = [[1,1,1,1]]#, [2,1,2,1], [2,2,1,1]]
    pendulalist = [[random.randint(2, 10)/2 for _ in range(4)] for _ in range(16)]
    upphafsstodur = [["π/3", 0, "π/6", 0], ["π/2", 0, "π/2", 0], ["π/12", 0, "-π/12", 0]]
    iterations = len(pendulalist) * len(upphafsstodur)
    counter = 0
    results = []
    print(f"starting with {iterations=}")
    t1 = time.time()
    for pendular in pendulalist:
        for upphafsstada in upphafsstodur:
            result_intermed = []
            for i in range(n_to_power_2):
                p = Pendulum(L_1=pendular[0], m_1=pendular[1], L_2=pendular[2], m_2=pendular[3])
                if i > 6:
                    n = 20000
                else:
                    n = 100*2**i
                th1, th2, thp1, thp2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=pi_[upphafsstada[0]], horn2=pi_[upphafsstada[2]],
                                      hornhradi1=upphafsstada[1], hornhradi2=upphafsstada[3], fjoldiskrefa=n, lengd=T, sp9=True)


                hnit1 = p.hornTohnit(th1[-1])
                hnit2 = p.hornTohnitjunior(th1[-1], th2[-1])
                # result_intermed.append([n, y1[-1], y2[-1], pendular, upphafsstada])
                result_dict = {"n": n, "th1": th1[-1], "th2": th2[-1], "thp1": thp1[-1], "thp2": thp2[-1], "pendular":pendular, "upphafsstada": upphafsstada}
                result_intermed.append(result_dict)
            results.append(result_intermed)
            counter+=1
            print(f"{counter/ iterations* 100:.00f}%", end="\n", flush=True)
    print(f"total time: {time.time() - t1:.02f}")
    for result in results:
        print(result)
    #np.savez("results_from_sp9.npz", results=results)

    # todo: plotta feril á pendulum með mismunandi n



def frjals(plot=False):
    pass


if __name__ == '__main__':
    # spurning1(plot=False)
    # spurning2(plot=False)
    # spurning3(plot=False)
    # spurning4(plot=False)
    # spurning5(plot=False)
    # spurning6(plot=False)
    # spurning7(plot=True)
    # exit()
    # spurning8(plot=False)
    spurning9(plot=True)
    # spurning12(plot=False)
    # frjals(plot=False)
