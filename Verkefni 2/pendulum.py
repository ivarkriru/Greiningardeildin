import random
import time
import numpy as np
import matplotlib.pyplot as plt
from adferdir import Foll, Pendulum
import math

def point_diff(A, B):
    return np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

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
        p.create_animation2d(hnit, title = "sp:3 Pendulum, theta(0) is pi/12, theta'(0) is 0 with Euler")

def spurning4(plot=False):
    p = Pendulum()
    hnit, y = p.hnitforanimationusingEuler(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)
    hnit2, y2 = p.hnitforanimationusingEuler(fall=p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=20)
    if plot:
        plt.plot(y)
        plt.plot(y2)
        plt.pause(2)
        plt.clf()
        p.create_animation2dfyrir4(data1 = hnit,data2=hnit2, title = "Sp4: Pendulums, blue theta(0): pi/12, green theta(0): pi/2, theta'(0): 0 with Euler")

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
    p = Pendulum(L_1=2, m_1=1, L_2=2, m_2=1)
    hnitsenior, hnitjunior, y1, y2 = p.hnitforanimationusingRK2(L_1=2, m_1=1, L_2=2, m_2=1, horn1=np.pi * 3 / 4,
                                  horn2=np.pi * 6 / 4,
                                  hornhradi1=0, hornhradi2=0, fjoldiskrefa=100, lengd=20)
    if plot:
        plt.plot(y1)
        plt.plot(y2)
        plt.show()
        p.create_animation2d(hnitsenior, hnitjunior, 2, trace=False, title="Sp7: Tvöfaldur pendúll,theta1 = pi/3, theta2 = pi/6, theta' = 0")

def spurning8(plot=False):
    pi_ = {"π/3": np.pi / 3, "π/6": np.pi / 6, "π/2": np.pi / 2, "π": np.pi, "π/4": np.pi / 4,"3*π/4": np.pi* 3/ 4, "6*π/4": np.pi* 6 / 4, 0: 0}
    def runspurning8(L_1=2, m_1=1, L_2=2, m_2=1, horn1=np.pi /2, horn2=np.pi /2, hornhradi1=1, hornhradi2=0, fjoldiskrefa=100, lengd=100):
        p= Pendulum()
        hnitsenior, hnitjunior, y1, y2 = p.hnitforanimationusingRK2(L_1=L_1, m_1=m_1, L_2=L_2, m_2=m_2, horn1= horn1,
                                      horn2= horn2, hornhradi1= hornhradi1, hornhradi2= hornhradi2, fjoldiskrefa= fjoldiskrefa, lengd= lengd)
        if plot:
            #print("here")
            #plt.plot(y1)
            #plt.plot(y2)
            #plt.title("")
            #plt.show()
            p.create_animation2d(hnitsenior, hnitjunior, 2,
                                 "Sp8: Tvöfaldur pendúll, theta1= "+
                                 str(list(pi_.keys())[list(pi_.values()).index(horn1)]) +
                                 ", theta2= "+ str(list(pi_.keys())[list(pi_.values()).index(horn2)])  + ",\ntheta1'= " + str(hornhradi1) + ", theta2'= " + str(hornhradi2) + ", lengd 1= "+ str(L_1)+ ", lengd 2= "+ str(L_2) +", þyngd 1= "+ str(m_1)+ ", þyngd 2= "+ str(m_2))

    fjoldiskrefa = 50
    lengd = 50
    #Gefin gildi:
    #L_1=2, m_1=1, L_2=2, m_2=1, horn1=np.pi /2, horn2=np.pi /2, hornhradi1=1, hornhradi2=0
    #Áhrif breytinga á theta1
    runspurning8(horn1=np.pi,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(horn1=np.pi/2,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(horn1=np.pi/4,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(horn1=0,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á theta2
    runspurning8(horn2=np.pi, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(horn1=np.pi/4,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á l1
    runspurning8(l_1 = 1, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(l_1 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á l2
    runspurning8(l_2 = 1, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(l_2 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á m1
    runspurning8(m_1 = 2, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(m_1 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á m2
    runspurning8(m_2 = 2, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(L_2 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á m2
    runspurning8(horn1 = np.pi, m_1=10, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(horn1 = np.pi, m_2=10, fjoldiskrefa=fjoldiskrefa, lengd=lengd)

pi_= {"π/3":np.pi/3, "π/6":np.pi/6, "π/2":np.pi/2, "π":np.pi, "π/4":np.pi/4, 0:0, "π/12":np.pi/12, "-π/12": -np.pi/12}
def spurning9(plot=False):
    follin = Foll()
    p = Pendulum()
    T = 20
    n_to_power_2 = 8  # 0-7
    #pendulalist = [[1,1,1,1]]#, [2,1,2,1], [2,2,1,1]]
    pendulalist = [[random.randint(1, 10)/2 for _ in range(4)] for _ in range(6)]
    upphafsstodur = [["π/3", 0, "π/6", 0]]# , ["π/2", 0, "π/2", 0], ["π/12", 0, "-π/12", 0]]
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
                    n = 100*2**i
                else:
                    n = 100*2**i
                array = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=pi_[upphafsstada[0]], horn2=pi_[upphafsstada[2]],
                                      hornhradi1=upphafsstada[1], hornhradi2=upphafsstada[3], fjoldiskrefa=n, lengd=T)

                result_dict = {"n": n, "th1": array[-1][0], "th2": array[-1][1], "thp1": array[-1][2], "thp2": array[-1][3], "pendular":pendular, "upphafsstada": upphafsstada}
                result_intermed.append(result_dict)
            results.append(result_intermed)
            counter+=1
            print(f"{counter/ iterations* 100:.00f}%", end="\n", flush=True)
    print(f"total time: {time.time() - t1:.02f}")
    for result in results:
        print(result)
    #np.savez("results_from_sp9.npz", results=results)

    #import json
    #file = open('results2.json', 'r')
    #results = json.load(file)
    #for result in results:
    #    print(result)
    # todo: plotta feril á pendulum með mismunandi n
    if plot:
        list_of_hallatales = []
        difffig, diffax = plt.subplots(1)
        fig, ax = plt.subplots(len(pendulalist), len(upphafsstodur), figsize=(10,6), facecolor=(.94, .94, .94))
        ax = np.asarray(ax).ravel()
        n_list = [result['n'] for result in results[0]]
        for index, result in enumerate(results):
            p = Pendulum(L_1=result[-1]['pendular'][0], L_2=result[-1]['pendular'][2],m_1=result[-1]['pendular'][1], m_2=result[-1]['pendular'][3])
            reasonable_coordinate = p.hornTohnitjunior(result[-1]['th1'], result[-1]['th2'], )
            x1 = [result_['th1'] for result_ in result[:-1]]
            y1 = [result_['th2'] for result_ in result[:-1]]
            # theta2 = [result_[2] for result_ in result]
            # hnit1 = [p.hornTohnit(result_[1], L_1=result_[3][0]) for result_ in result]
            hnit2_list = [p.hornTohnitjunior(result_['th1'], result_['th2']) for result_ in result[:-1]]

            diff = [point_diff(hnit2, reasonable_coordinate) for hnit2 in hnit2_list]
            print(n_list)
            print(diff)
            diffax.plot(np.log(n_list[:-1]), np.log(diff))
            list_of_hallatales.append(np.polyfit(np.log(n_list[:-1]), np.log(diff), 1)[0])
            #ax[index].plot([xy[0] for xy in hnit2_list], [xy[1] for xy in hnit2_list])
            #for i in range(len(hnit2_list)):
            #    ax[index].text(hnit2_list[i][0]+0.01, hnit2_list[i][1]+0.01, i)
            #ax[index].set_xlim([-10.1, 10.1])
            #ax[index].set_ylim([-10.1, 10.1])

            # x = [result_[0]+50 for result_ in result] # til að bars séu ekki ofan í hvorum öðrum
            # ax[index].bar(x, theta2, 100, color='red')
            #ax[index].set_title(f"{result[0]['upphafsstada']}, L1: {result[0]['pendular']}")
            # ax.bar(x, theta2)
        plt.figure()
        plt.hist(list_of_hallatales)
        print(f"Average of hallatales: {np.average(list_of_hallatales)}, mean: {np.mean(list_of_hallatales)}")
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
    #spurning1(plot=True)
    #spurning2(plot=True)
    #spurning3(plot=True)
    #spurning4(plot=True)
    #spurning5(plot=True)
    #spurning6(plot=True)
    spurning7(plot=True)
    # spurning8(plot=False)
    #spurning9(plot=True)
    # spurning10(plot=False)
    # spurning11(plot=False)
    # spurning12(plot=False)
    # frjals(plot=False)
