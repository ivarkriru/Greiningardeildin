import numpy as np
import matplotlib.pyplot as plt
from adferdir import Foll, Pendulum
import math


def point_diff(A, B):
    return np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2)


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


def spurning3(plot=False):
    p = Pendulum()
    hnit, y = p.hnitforanimationusingEuler(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)
    if plot:
        plt.plot(y)
        plt.pause(2)
        plt.clf()
        p.create_animation2d(hnit, title="Pendulum, theta(0) is pi/12, theta'(0) is 0 with Euler")


def spurning4(plot=False):
    p = Pendulum()
    hnit, y = p.hnitforanimationusingEuler(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)
    hnit2, y2 = p.hnitforanimationusingEuler(fall=p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=20)
    if plot:
        plt.plot(y)
        plt.plot(y2)
        plt.pause(2)
        plt.clf()
        p.create_animation2dfyrir4(data1=hnit, data2=hnit2,
                                   title="Pendulums, blue theta(0): pi/12, green theta(0): pi/2, theta'(0): 0 with Euler")


def spurning5(plot=False):
    p = Pendulum()
    hnit, y = p.hnitforanimationusingRK(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)
    hnit2, y2 = p.hnitforanimationusingRK(fall=p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=20)

    if plot:
        plt.plot(y)
        plt.plot(y2)
        plt.pause(2)
        plt.clf()
        p.create_animation2dfyrir4(data1=hnit, data2=hnit2,
                                   title="Pendulums, blue theta(0): p/12, green theta(0): p/2, theta'(0): 0 with RK")


def spurning6(plot=False):
    '''

    er á ipad pdf skjalinu hér á git.

    '''
    pass


def spurning7(plot=False):
    follin = Foll()
    p = Pendulum(L_1=2, m_1=1, L_2=2, m_2=1)
    lengdin = 100
    y1, y2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=np.pi * 3 / 4, horn2=np.pi * 6 / 4,
                              hornhradi1=1, hornhradi2=0, fjoldiskrefa=lengdin * 30, lengd=lengdin)
    hnitsenior = []
    hnitjunior = []

    for theta in y1:
        hnitsenior.append(p.hornTohnit(theta))
    for index, theta in enumerate(y2):
        hnitjunior.append(p.hornTohnitjunior(y1[index], theta))

    hnitsenior = np.array(hnitsenior)
    hnitjunior = np.array(hnitjunior)

    if plot:
        plt.plot(y1)
        plt.plot(y2)
        plt.show()
        p.create_animation2d(hnitsenior, hnitjunior, 2)


def spurning8(plot=False):
    follin = Foll()
    p = Pendulum(L_1=4, m_1=4, L_2=4, m_2=4)
    lengdin = 100
    y1, y2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=np.pi / 2, horn2=np.pi / 2,
                              hornhradi1=0, hornhradi2=0, fjoldiskrefa=lengdin * 30, lengd=lengdin)
    hnitsenior = []
    hnitjunior = []
    for theta in y1:
        hnitsenior.append(p.hornTohnit(theta))
    for index, theta in enumerate(y2):
        hnitjunior.append(p.hornTohnitjunior(y1[index], theta))

    hnitsenior = np.array(hnitsenior)
    hnitjunior = np.array(hnitjunior)

    if plot:
        plt.plot(y1)
        plt.plot(y2)
        plt.show()
        p.create_animation2d(hnitsenior, hnitjunior, 2)



pi_= {"π/3":np.pi/3, "π/6":np.pi/6, "π/2":np.pi/2, "π":np.pi, "π/4":np.pi/4, 0:0}
def spurning9(plot=False):
    follin = Foll()
    p = Pendulum(L_1=22, m_1=11, L_2=5, m_2=107)
    T = 20
    pendulalist = [[1,1,1,1]]# , [2,1,2,1], [2,2,1,1]]
    upphafsstodur = [["π/3", 0, "π/6", 0], ["π/2", 0, "π", 0], ["π/2", 0, 0, 0]]
    results = []
    for pendular in pendulalist:
        for upphafsstada in upphafsstodur:
            result_intermed = []
            for i in range(6+2):
                p = Pendulum(L_1=pendular[0], m_1=pendular[1], L_2=pendular[2], m_2=pendular[3])
                if i > 6:
                    n = 20000
                else:
                    n = 100*2**i

                th1, th2, thp1, thp2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=pi_[upphafsstada[0]], horn2=pi_[upphafsstada[2]],
                                      hornhradi1=upphafsstada[1], hornhradi2=upphafsstada[3], fjoldiskrefa=n, lengd=20, sp9=True)


                hnit1 = p.hornTohnit(th1[-1])
                hnit2 = p.hornTohnitjunior(th1[-1], th2[-1])
                # result_intermed.append([n, y1[-1], y2[-1], pendular, upphafsstada])
                result_dict = {"n": n, "th1": th1[-1], "th2": th2[-1], "thp1": thp1[-1], "thp2": thp2[-1], "pendular":pendular, "upphafsstada": upphafsstada}
                result_intermed.append(result_dict)
            results.append(result_intermed)
    for result in results:
        print(result)

    # todo: plotta feril á pendulum með mismunandi n
    if plot:


        fig, ax = plt.subplots(len(pendulalist), len(upphafsstodur), figsize=(10,6), facecolor=(.94, .94, .94))
        ax = np.asarray(ax).ravel()
        for index, result in enumerate(results):
            # x = [result_[0] for result_ in result]
            x1 = [result_[2][0] for result_ in result]
            y1 = [result_[2][1] for result_ in result]
            # theta2 = [result_[2] for result_ in result]
            # hnit1 = [p.hornTohnit(result_[1], L_1=result_[3][0]) for result_ in result]
            # hnit2 = [p.hornTohnitjunior(result_[1], result[2], L_1=result[3][0], L_2=result_[3][2]) for result_ in result]


            ax[index].plot(x1, y1)
            ax[index].set_xlim([np.min(x1)-.1,np.max(x1)+.1])
            ax[index].set_ylim([np.min(y1)-.1,np.max(y1)+.1])


            # x = [result_[0]+50 for result_ in result] # til að bars séu ekki ofan í hvorum öðrum
            # ax[index].bar(x, theta2, 100, color='red')
            ax[index].set_title(f"{result[0][4]}, L1: {result[0][3]}")
            # ax.bar(x, theta2)
        plt.show()


def spurning10(plot=False):
    follin = Foll()
    p = Pendulum(L_1=2, m_1=1, L_2=2, m_2=2)
    lengdin = 1000
    y1, y2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=np.pi / 7, horn2=np.pi * 1.5,
                              hornhradi1=-6, hornhradi2=10, fjoldiskrefa=lengdin * 20, lengd=lengdin)
    '''
    for i,x in enumerate(y1):
        y1[i] = x%np.pi
    for i,x in enumerate(y2):
        y2[i] = x%np.pi
    '''
    if plot:
        plt.plot(y1, y2)
        plt.show()


def spurning11(plot=False):
    follin = Foll()
    p = Pendulum()
    lengdin = 40
    for x in [1, 2, 3, 4, 5]:
        epsilon = math.pow(10, -1 * x)
        y1, y2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=np.pi * 2 / 3, horn2=np.pi / 6,
                                  hornhradi1=0, hornhradi2=0, fjoldiskrefa=lengdin * 30, lengd=lengdin)

        y3, y4 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=np.pi * 2 / 3 + epsilon,
                                  horn2=np.pi / 6 + epsilon,
                                  hornhradi1=0, hornhradi2=0, fjoldiskrefa=lengdin * 30, lengd=lengdin)
        hnit1 = []
        hnit2 = []
        hnit3 = []
        hnit4 = []

        for theta in y1:
            hnit1.append(p.hornTohnit(theta))
        for index, theta in enumerate(y2):
            hnit2.append(p.hornTohnitjunior(y1[index], theta))

        for theta in y3:
            hnit3.append(p.hornTohnit(theta))
        for index, theta in enumerate(y4):
            hnit4.append(p.hornTohnitjunior(y3[index], theta))

        hnit1 = np.array(hnit1)
        hnit2 = np.array(hnit2)
        hnit3 = np.array(hnit3)
        hnit4 = np.array(hnit4)

        if plot:
            plt.plot(y1)
            plt.plot(y2)

            plt.plot(y3)
            plt.plot(y4)
            plt.show()
            p.create_animation2ex2(hnit1, hnit2, hnit3, hnit4)


def spurning12(plot=False):
    follin = Foll()
    p = Pendulum()

    # breyta eftirfarandi og skoða áhrifin

    # skoða breytingu á lengd tímabila
    lengdin = 40
    # nákvæmni gildanna
    n = 30
    nakvaemni = lengdin * n
    # upphafsgildi theta 1 og theta 2
    theta1 = np.pi * 2 / 3
    theta2 = np.pi / 6

    for x in [*range(1, 13)]:
        epsilon = math.pow(10, -1 * x)
        y1, y2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=theta1, horn2=theta2,
                                  hornhradi1=0, hornhradi2=0, fjoldiskrefa=lengdin * 30, lengd=lengdin)

        y3, y4 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=np.pi * 2 / 3 + epsilon,
                                  horn2=np.pi / 6 + epsilon,
                                  hornhradi1=0, hornhradi2=0, fjoldiskrefa=nakvaemni, lengd=lengdin)
        hnit1 = []
        hnit2 = []
        hnit3 = []
        hnit4 = []

        for theta in y1:
            hnit1.append(p.hornTohnit(theta))
        for index, theta in enumerate(y2):
            hnit2.append(p.hornTohnitjunior(y1[index], theta))

        for theta in y3:
            hnit3.append(p.hornTohnit(theta))
        for index, theta in enumerate(y4):
            hnit4.append(p.hornTohnitjunior(y3[index], theta))

        hnit1 = np.array(hnit1)
        hnit2 = np.array(hnit2)
        hnit3 = np.array(hnit3)
        hnit4 = np.array(hnit4)

        if plot:
            plt.plot(y1)
            plt.plot(y2)

            plt.plot(y3)
            plt.plot(y4)
            plt.show()
            p.create_animation2ex2(hnit1, hnit2, hnit3, hnit4)


def frjals(plot=False):
    pass


if __name__ == '__main__':
    spurning1(plot=False)
    spurning2(plot=False)
    spurning3(plot=False)
    spurning4(plot=False)
    spurning5(plot=False)
    spurning6(plot=False)
    spurning7(plot=False)
    spurning8(plot=False)
    spurning9(plot=False)
    spurning10(plot=False)
    spurning11(plot=True)
    # spurning12(plot=False)
    # frjals(plot=False)
