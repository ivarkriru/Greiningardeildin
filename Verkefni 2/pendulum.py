import numpy as np
import matplotlib.pyplot as plt
from adferdir import Foll, Pendulum


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
    hnit,y = p.hnitforanimationusingEuler(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)
    if plot:
        plt.plot(y)
        plt.show()
        plt.pause(2)
        p.create_animation2d(hnit, title = "Pendulum, theta(0) is p/12, theta'(0) is 0 with Euler")



def spurning4(plot=False):
    p = Pendulum()
    hnit,y = p.hnitforanimationusingEuler(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)
    hnit2,y2 = p.hnitforanimationusingEuler(fall=p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=20)
    if plot:
        plt.plot(y)
        plt.plot(y2)
        plt.show()
        plt.pause(4)
        p.create_animation2dfyrir4(data1 = hnit,data2=hnit2, title = "Pendulums, blue theta(0): p/12, green theta(0): p/2, theta'(0): 0 with Euler")


def spurning5(plot=False):
    p = Pendulum()
    hnit,y = p.hnitforanimationusingRK(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)
    hnit2,y2 = p.hnitforanimationusingRK(fall=p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=20)

    if plot:
        plt.plot(y)
        plt.plot(y2)
        plt.show()
        plt.pause(4)
        p.create_animation2dfyrir4(data1 = hnit, data2=hnit2, title = "Pendulums, blue theta(0): p/12, green theta(0): p/2, theta'(0): 0 with RK")



def spurning6(plot=False):
    '''

    er á ipad pdf skjalinu hér á git.

    '''
    pass


def spurning7(plot=False):
    follin = Foll()
    p = Pendulum(L_1=1, m_1=1, L_2=0.5, m_2=1)
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




def spurning9(plot=False):
    follin = Foll()
    p = Pendulum(L_1=22, m_1=11, L_2=5, m_2=107)
    T = 20
    pendulalist = [[1,1,1,1], [2,1,2,1], [2,2,1,1]]
    upphafsstodur = [[np.pi/3, 0, np.pi/6, 0], [np.pi/2, 0, np.pi, 0]]
    results = []
    for pendular in pendulalist:
        for upphafsstada in upphafsstodur:
            result_intermed = []
            for i in range(6+1):
                p = Pendulum(L_1=pendular[0], m_1=pendular[1], L_2=pendular[2], m_2=pendular[3])
                n = 100*2**i
                y1, y2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=upphafsstada[0], horn2=upphafsstada[2],
                                      hornhradi1=upphafsstada[1], hornhradi2=upphafsstada[3], fjoldiskrefa=n, lengd=20)
                result_intermed.append([n, y1[-1], y2[-1], pendular, upphafsstada])
            results.append(result_intermed)
    for result in results:
        print(result)

    # todo: plotta feril á pendulum með mismunandi n
    if plot:
        fig, ax = plt.subplots(len(pendulalist)*len(upphafsstodur), figsize=(10,6), facecolor=(.94, .94, .94))
        for index, result in enumerate(results):
            x = [result_[0] for result_ in result]
            theta1 = [result_[1] for result_ in result]
            theta2 = [result_[2] for result_ in result]

            #plt.show()
            print(x)
            print(theta1)
            ax[index].bar(x, theta1, 500)
            ax[index].bar(x, theta2, 500, color='red')
            #ax.bar(x, theta2)
        plt.show()

def spurning10(plot=False):
    pass


def spurning11(plot=False):
    pass


def spurning12(plot=False):
    pass


def frjals(plot=False):
    pass


if __name__ == '__main__':
    spurning1()
    spurning2()
    spurning3()
    spurning4()
    spurning5()
    spurning6()
    spurning7()
    #spurning8(plot=True)
    spurning9(plot=True)
