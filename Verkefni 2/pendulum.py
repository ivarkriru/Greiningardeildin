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
    hnit = p.hnitforanimation(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)
    if plot:
        p.create_animation2d(hnit)


def spurning4(plot=False):
    p = Pendulum()
    hnit = p.hnitforanimation(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)
    hnit2 = p.hnitforanimation(fall=p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=20)
    if plot:
        p.create_animation2d(data1 = hnit, title = "Pendulums, theta(0) is p/12, theta'(0) is p/12")


def spurning5(plot=False):
    follin = Foll()
    p = Pendulum()
    y = follin.RKmethod(f=p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=10)
    hnit = []
    for theta in y:
        hnit.append(p.hornTohnit(theta))
    hnit = np.array(hnit)

    if plot:
        p.create_animation2dfyrir4(data1 = hnit, data2=hnit2, title = "Pendulums, for green theta is pi/2 and for blue theta is p/12")



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
    pass


def spurning10(plot=False):
    follin = Foll()
    p = Pendulum(L_1=22, m_1=11, L_2=5, m_2=107)
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
    pass


def spurning12(plot=False):
    pass


def frjals(plot=False):
    pass


if __name__ == '__main__':
    spurning1()
    spurning2()
    spurning3(True)
    spurning4(True)
    spurning5()
    spurning6()
    spurning7()
    # spurning8(plot=True)
    # spurning10()
