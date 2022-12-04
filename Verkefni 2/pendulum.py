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

        plt.title(r"Sp:3 Graf yfir pendúll $\dot{\Theta}$(0) er $\dot{\pi}$/12, $\dot{\Theta}$'(0) er 0 með Euler")
        plt.xlabel('Fjöldi skrefa í aðferð Eulers')
        plt.ylabel('Horn [°] pendúls')
        plt.pause(2)
        plt.clf()
        p.create_animation2d(hnit, title = r"Sp:3 Einfaldur pendúll, $\dot{\Theta}$(0) er pi/12, $\dot{\Theta}$'(0) er 0 með Euler")
        plt.clf()


def spurning4(plot=False):
    p = Pendulum()

    hnit, y = p.hnitforanimationusingEuler(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20)
    hnit2, y2 = p.hnitforanimationusingEuler(fall=p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=20)
    if plot:
        plt.plot(y)
        plt.plot(y2)
        plt.xlabel('Fjöldi skrefa í aðferð Eulers')
        plt.ylabel('Horn [°] pendúls')
        plt.title(r"Sp:4 Graf yfir pendúla blár er $\dot{\Theta}$(0) er $\dot{\pi}$/12, grænn er $\dot{\pi}$/2"+ "\n" + r"$\dot{\Theta}$'(0) er 0 með Euler")
        plt.pause(2)
        plt.clf()
        p.create_animation2dfyrir4(data1 = hnit,data2=hnit2, title = r"Sp4: Pendúlar, blár er $\dot{\Theta}$(0) er $\dot{\pi}$/12, grænn er $\dot{\pi}$/2"+ "\n" + r"$\dot{\Theta}$'(0) er 0 með Euler")
        plt.clf()

def spurning5(plot=False):
    p = Pendulum()
    hnit, y = p.hnitforanimationusingRK(fall=p.pendulum, horn=np.pi / 12, hornhradi=0, fjoldiskrefa=500, lengd=20,dempunarstuðull=0.1)
    hnit2, y2 = p.hnitforanimationusingRK(fall=p.pendulum, horn=np.pi / 2, hornhradi=0, fjoldiskrefa=500, lengd=20)

    if plot:
        plt.plot(y)
        plt.plot(y2)
        plt.xlabel('Fjöldi skrefa í aðferð RK')
        plt.ylabel('Horn [°] pendúls')
        plt.title(r"Sp:5 Graf yfir pendúla, blár er $\dot{\Theta}$(0)= $\dot{\pi}$/12, grænn er $\dot{\Theta}$(0)= $\dot{\pi}$/2"+ "\n" + r"$\dot{\Theta}$'(0)= 0 með RK")
        plt.pause(2)
        plt.clf()
        p.create_animation2dfyrir4(data1 = hnit, data2=hnit2, title = r"Sp5: Pendúlar, blár er $\dot{\Theta}$(0)= $\dot{\pi}$/12, grænn er $\dot{\Theta}$(0)= $\dot{\pi}$/2"+ "\n" + r"$\dot{\Theta}$'(0)= 0 með RK")
        plt.clf()

def spurning6(plot=False):
    '''

    er á ipad pdf skjalinu hér á git.

    '''
    pass

def spurning7(plot=False):
    p = Pendulum(L_1=2, m_1=1, L_2=2, m_2=1)
    hnitsenior, hnitjunior, y1, y2 = p.hnitforanimationusingRK2(L_1=2, m_1=1, L_2=2, m_2=1, horn1=np.pi ,
                                  horn2=np.pi/2,
                                  hornhradi1=0, hornhradi2=0, fjoldiskrefa=10, lengd=20,dempunarstuðull=0.5)
    if plot:
        plt.clf()
        plt.plot(y1)
        plt.plot(y2)
        plt.xlabel('Fjöldi skrefa í aðferð RK')
        plt.ylabel('Horn [°] pendúls')
        plt.title(r"Sp7: Tvöfaldur pendúll,blár er $\dot{\Theta}$1(0) = $\dot{\pi}$/3, appelsínugulur er $\dot{\Theta}$2(0) = $\dot{\pi}$/6,"+ "\n" + r"$\dot{\Theta}$' = 0")
        plt.pause(2)
        plt.clf()

        p.create_animation2d(hnitsenior, hnitjunior, 2, trace=False, title=r"Sp7: Tvöfaldur pendúll,blár er $\dot{\Theta}$1(0) = $\dot{\pi}$/3, grænn er $\dot{\Theta}$2(0) = $\dot{\pi}$/6,"+ "\n" + r"$\dot{\Theta}$' = 0")
        plt.clf()

pi_= {"π/3":np.pi/3, "π/6":np.pi/6, "π/2":np.pi/2, "π":np.pi, "π/4":np.pi/4, 0:0, "π/12":np.pi/12, "-π/12": -np.pi/12}
def spurning8(plot=False):
    def runspurning8(L_1=2, m_1=1, L_2=2, m_2=1, horn1=np.pi /2, horn2=np.pi /2, hornhradi1=1, hornhradi2=0, fjoldiskrefa=10, lengd=100):
        p= Pendulum()
        hnitsenior, hnitjunior, y1, y2 = p.hnitforanimationusingRK2(L_1=L_1, m_1=m_1, L_2=L_2, m_2=m_2, horn1= horn1,
                                      horn2= horn2, hornhradi1= hornhradi1, hornhradi2= hornhradi2, fjoldiskrefa= fjoldiskrefa, lengd= lengd)
        if plot:
            horn1 = str(list(pi_.keys())[list(pi_.values()).index(horn1)])
            horn2 = str(list(pi_.keys())[list(pi_.values()).index(horn2)])
            plt.plot(y1)
            plt.plot(y2)
            plt.xlabel('Fjöldi skrefa í aðferð RK')
            plt.ylabel('Horn [°] pendúls')
            plt.title(r"Sp8: Graf af pendúlum, blár er $\dot{\Theta}$1(0) ="+
                                 str(horn1) +
                                 r", grænn er $\dot{\Theta}$2(0) ="+ str(horn2)  + ",\ntheta1'= " + str(hornhradi1) + ", theta2'= " + str(hornhradi2) + ", lengd 1= "+ str(L_1)+ ", lengd 2= "+ str(L_2) +", þyngd 1= "+ str(m_1)+ ", þyngd 2= "+ str(m_2))
            plt.pause(2)
            plt.clf()
            p.create_animation2d(hnitsenior, hnitjunior, 2,
                                 r"Sp8:, blár er $\dot{\Theta}$1(0) ="+
                                 str(horn1) +
                                 r", grænn er $\dot{\Theta}$2(0) ="+ str(horn2)  + ",\ntheta1'= " + str(hornhradi1) + ", theta2'= " + str(hornhradi2) + ", lengd 1= "+ str(L_1)+ ", lengd 2= "+ str(L_2) +", þyngd 1= "+ str(m_1)+ ", þyngd 2= "+ str(m_2))
            plt.clf()

    #title = r"Sp7: Tvöfaldur pendúll,blár er $\dot{\Theta}$1(0) = $\dot{\pi}$/3r", grænn er $\dot{\Theta}$2(0) =" $\dot{\pi}$/6," + "\n" + r"$\dot{\Theta}$' = 0")

    fjoldiskrefa = 100
    lengd = 20
    runspurning8(horn1=np.pi,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(horn1=np.pi/2,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(horn1=np.pi/4,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(horn1=0,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á theta2
    runspurning8(horn2=np.pi, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(horn1=np.pi/4,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á l1
    runspurning8(L_1 = 1, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(L_1 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á l2
    runspurning8(L_2 = 1, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(L_2 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á m1
    runspurning8(m_1 = 2, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(m_1 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á m2
    runspurning8(m_2 = 2, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(L_2 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á m2
    runspurning8(horn1 = np.pi, m_1=10, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning8(horn1 = np.pi, m_2=10, fjoldiskrefa=fjoldiskrefa, lengd=lengd)

def spurning9(plot=False):
    follin = Foll()
    T = 20
    n_to_power_2 = 8  # 0-7
    fjoldi_uppsetninga = 15
    #pendulalist = [[1,1,1,1]]#, [2,1,2,1], [2,2,1,1]]
    pendulalist = [[random.randint(1, 10)/2 for _ in range(4)] for _ in range(fjoldi_uppsetninga)]
    upphafsstodur = [["π/6", 0, "π/6", 0], ["π/2", 0, "π/12", 0], ["π/12", 0, "-π/12", 0]]
    iterations = len(pendulalist) * len(upphafsstodur)
    counter = 0
    results = []
    print(f"starting with {iterations=}")
    t1 = time.time()
    time_vectors = []
    for pendular in pendulalist:
        for upphafsstada in upphafsstodur:
            result_intermed = []
            for i in range(n_to_power_2):
                p = Pendulum(L_1=pendular[0], m_1=pendular[1], L_2=pendular[2], m_2=pendular[3])
                if i > 6:
                    n = 30000
                else:
                    n = 100*2**i
                array = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=pi_[upphafsstada[0]], horn2=pi_[upphafsstada[2]],
                                      hornhradi1=upphafsstada[1], hornhradi2=upphafsstada[3], fjoldiskrefa=n, lengd=T)

                result_dict = {"n": n, "th1": array[-1][0], "th2": array[-1][1], "thp1": array[-1][2], "thp2": array[-1][3], "pendular":pendular, "upphafsstada": upphafsstada}
                result_intermed.append(result_dict)
            results.append(result_intermed)
            counter+=1
            print(f"{counter/ iterations* 100:.00f}%", end="\n", flush=True)
            time_vectors.append((time.time() - t1) / counter)
            estimated_total_time = iterations * np.average(time_vectors)
            estimated_time_left = estimated_total_time - sum(time_vectors)
            print(f"estimated time left: {estimated_time_left:.00f}")
    print(f"total time: {time.time() - t1:.02f}")
    results = np.load("results_from_sp9_1669987822.npz", allow_pickle=True)['results']
    print(results)
    #np.savez(f"results_from_sp9_{time.time():.00f}.npz", results=results)


    #import json
    #file = open('results2.json', 'r')
    #results = json.load(file)
    #for result in results:
    #    print(result)
    # todo: plotta feril á pendulum með mismunandi n
    if plot:
        list_of_hallatales = []
        difffig, diffax = plt.subplots(1)
        #fig, ax = plt.subplots(len(pendulalist), len(upphafsstodur), figsize=(10,6), facecolor=(.94, .94, .94))
        #ax = np.asarray(ax).ravel()
        n_list = [result['n'] for result in results[0]]
        for index, result in enumerate(results):
            p = Pendulum(L_1=result[-1]['pendular'][0], L_2=result[-1]['pendular'][2],m_1=result[-1]['pendular'][1], m_2=result[-1]['pendular'][3])
            reasonable_coordinate = p.hornTohnitjunior(result[-1]['th1'], result[-1]['th2'], )
            x1 = [result_['th1'] for result_ in result[:-1]]
            y1 = [result_['th2'] for result_ in result[:-1]]
            hnit2_list = [p.hornTohnitjunior(result_['th1'], result_['th2']) for result_ in result[:-1]]
            theta1_best = result[-1]['th1']
            theta2_best = result[-1]['th2']
            thp1_best = result[-1]['thp1']
            thp2_best = result[-1]['thp2']
            # diff = [point_diff(hnit2, reasonable_coordinate) for hnit2 in hnit2_list]
            # (theta2-theta1)(thetaprime2 - thetaprime1)(thetabest1-thetabest2)
            #diff = [math.sqrt((theta1_best-result_['th1'])**2 + (theta2_best-result_['th2'])**2 + (thp1_best-result_['thp1'])**2 + (thp2_best-result_['thp2'])**2) for result_ in result[:-1]]
            diff = [np.linalg.norm(np.abs([(result_['th1']-theta1_best), (result_['th2']-theta2_best), (result_['thp1']- thp1_best),  result_['thp2']-thp2_best])) for result_ in result[:-1]]

            if diff[0] > 100 or diff[1] > 100:
                print(f"throwing away {diff}")
                continue

            print(n_list)
            print(diff)

            #### uncommenta fyrir svar9c.png og svar9d.png #####

            #if diff[-1] > 1e-9:
            #    continue
            diffax.loglog(n_list[:-1], diff)
            list_of_hallatales.append(np.polyfit(np.log(n_list[:-1]), np.log(diff), 1)[0])

            # uncomment til að fá plot á lokastaðsetningum
            #plt.plot([xy[0] for xy in hnit2_list], [xy[1] for xy in hnit2_list])
            #for i in range(len(hnit2_list)):
            #    plt.text(hnit2_list[i][0]+0.0001, hnit2_list[i][1]+0.0001, i)
            #plt.xlabel("x")
            #plt.ylabel("y")
            #plt.show()

            #ax[index].set_xlim([-10.1, 10.1])
            #ax[index].set_ylim([-10.1, 10.1])

            # x = [result_[0]+50 for result_ in result] # til að bars séu ekki ofan í hvorum öðrum
            # ax[index].bar(x, theta2, 100, color='red')
            #ax[index].set_title(f"{result[0]['upphafsstada']}, L1: {result[0]['pendular']}")
            # ax.bar(x, theta2)
        plt.title("Sp9a: Hámarksskekkja eftir fjölda(n)")
        plt.xlabel("Fjöldi (n)")
        plt.ylabel("hámarks skekkja af θ_1, θ_2, θ'_1, θ'_2")
        plt.xticks([100, 200, 400, 800, 1600, 3200, 6400], [str(100*2**i) for i in range(7)])
        plt.savefig("svar9a.png")
        plt.figure()
        plt.hist(list_of_hallatales, bins=40)
        plt.title("Sp9b: Súlurit, fjöldi skekkja í hverjum stuðli á skekkju")
        plt.xlabel("Hallatala")
        plt.ylabel("Fjöldi")
        plt.savefig("svar9b.png")
        print(f"Average of hallatales: {np.average(list_of_hallatales)}, mean: {np.mean(list_of_hallatales)}")

        plt.show()
        #plt.title("Sp9: Súlurit, fjöldi skekkja í hverjum stuðli á skekkju")
        #plt.xlabel("Stuðull á skekkju")
        #plt.ylabel("Fjöldi skekkja")
        #plt.title("Sp9: Súlurit, fjöldi skekkja í hverjum stuðli á skekkju")

def spurning10(plot=False):
    def runspurning10(L_1=2, m_1=1, L_2=2, m_2=1, horn1=np.pi /3, horn2=np.pi /6, hornhradi1=0, hornhradi2=0, fjoldiskrefa=10, lengd=20):
        follin = Foll()
        p = Pendulum(L_1= L_1, m_1= m_1, L_2= L_2, m_2= m_2)
        arr = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=horn1 , horn2=horn2,
                              hornhradi1=hornhradi1, hornhradi2=hornhradi2, fjoldiskrefa=lengd * 1000, lengd=lengd)
        y1 = arr[:,0]
        y2 = arr[:,1]
        '''
        for i,x in enumerate(y1):
            y1[i] = x%np.pi
        for i,x in enumerate(y2):
            y2[i] = x%np.pi
        '''
        if plot:
            plt.plot(y1, y2)
            plt.xlabel('Staðsetning á x-ás í radíönum')
            plt.ylabel('Staðsetning á y-ás í radíönum')
            plt.title(r"Sp8:, blár= $\dot{\Theta}$1(0) ="+str(horn1)+r", grænn er $\dot{\Theta}$2(0) ="+ str(horn2)
                + ",\n" + r"$\dot{\Theta}$1'= " + str(hornhradi1) + r"$\dot{\Theta}$2'= " + str(hornhradi2) + ", lengd 1= "+ str(L_1)+ ", lengd 2= "+ str(L_2) +", þyngd 1= "+ str(m_1)+ ", þyngd 2= "+ str(m_2))
            plt.show()
            plt.pause(2)
            plt.clf()


    fjoldiskrefa = 100
    lengd = 20
    runspurning10(horn1=np.pi,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning10(horn1=np.pi/2,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning10(horn1=np.pi/4,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning10(horn1=0,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á theta2
    runspurning10(horn2=np.pi, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning10(horn1=np.pi/4,fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á l1
    runspurning10(L_1 = 1, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning10(L_1 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á l2
    runspurning10(L_2 = 1, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning10(L_2 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á m1
    runspurning10(m_1 = 2, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning10(m_1 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á m2
    runspurning10(m_2 = 2, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning10(L_2 = 3, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    #Áhrif breytinga á m2
    runspurning10(horn1 = np.pi, m_1=10, fjoldiskrefa=fjoldiskrefa, lengd=lengd)
    runspurning10(horn1 = np.pi, m_2=10, fjoldiskrefa=fjoldiskrefa, lengd=lengd)



def spurning11(plot=False):
    follin = Foll()
    p = Pendulum()
    lengdin = 20
    for x in [1, 2, 3, 4, 5]:
        epsilon = math.pow(10, -1 * x)
        arr1 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=np.pi * 2 / 3, horn2=np.pi / 6,
                                  hornhradi1=0, hornhradi2=0, fjoldiskrefa=lengdin * 1000, lengd=lengdin,dempunarstuðull=0)

        arr2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=np.pi * 2 / 3 + epsilon,
                                  horn2=np.pi / 6 + epsilon,
                                  hornhradi1=0, hornhradi2=0, fjoldiskrefa=lengdin * 1000, lengd=lengdin,dempunarstuðull=0)

        y1 = arr1[:,0]
        y2 = arr1[:,1]
        y3 = arr2[:,0]
        y4 = arr2[:,1]

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
    lengdin = 10
    # nákvæmni gildanna
    n = 1000
    # upphafsgildi theta 1 og theta 2
    theta1 = np.pi
    theta2 = np.pi /2
    hnit1 = []
    hnit2 = []
    hnit3 = []
    hnit4 = []
    hnit1_list = []
    hnit2_list = []
    for x in range(1, 13):
        print(x)
        epsilon = math.pow(10, -x)
        arr1 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=theta1, horn2=theta2,
                                  hornhradi1=0, hornhradi2=0, fjoldiskrefa=n, lengd=lengdin)

        arr2 = follin.RKmethod2(f1=p.double_pendulum1, f2=p.double_pendulum2, horn1=theta1 + epsilon,
                                  horn2=theta2 + epsilon,
                                  hornhradi1=0, hornhradi2=0, fjoldiskrefa=n, lengd=lengdin)
        y1 = arr1[:,0]
        y2 = arr1[:,1]
        y3 = arr2[:,0]
        y4 = arr2[:,1]
        hnit1 = []
        hnit2 = []
        hnit3 = []
        hnit4 = []

        hnit1_list.append(p.hornTohnitjunior(y1[-1], y2[-1]))
        hnit2_list.append(p.hornTohnitjunior(y3[-1], y4[-1]))

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
            pass
            # plt.plot(y1)
            # plt.plot(y2)

            # plt.plot(y3)
            # plt.plot(y4)
            # plt.show()
            #p.create_animation2ex2(hnit1, hnit2, hnit3, hnit4)
    if plot:
        diff = [point_diff(hnit1, hnit2) for hnit1, hnit2 in zip(hnit1_list, hnit2_list)]
        plt.plot([i for i in range(1,13)], diff)
        plt.xlabel("k")
        plt.ylabel("norm á milli punkta")
        plt.xticks([i for i in range(1,13)], [i for i in range(1,13)])
        plt.show()
def frjals(plot=False):
    p = Pendulum(L_1=3, m_1=1, L_2=2, m_2=3,L_3=3,m_3=1)
    lengd = 40
    hnitsenior, hnitjunior,hnitjuniorjunior, y1, y2, y3 = p.hnitforanimationusingRK3(horn1=np.pi+0.00001, horn2=np.pi-0.00001, horn3=np.pi,
                                                                hornhradi1=0, hornhradi2=0, hornhradi3=0,
                                                                fjoldiskrefa=1000*lengd, lengd=lengd, dempunarstuðull=0)
    if plot:
        plt.clf()
        plt.plot(y1)
        plt.plot(y2)
        plt.plot(y3)
        plt.xlabel('Fjöldi skrefa í aðferð RK')
        plt.ylabel('Horn [°] pendúls')
        plt.title(r"Þrefaldur pendúll")
        plt.pause(2)
        plt.clf()
        p.create_animation3d(hnitsenior, hnitjunior, hnitjuniorjunior)
        plt.clf()

""" 
ATH: ný leið til að keyra föll til að fækka conflicts: 

búa til skjal sem heitir main.py sem inniheldur: 
from pendulum import *

# spurning1(plot=False)
# spurning2(plot=False)
# spurning3(plot=False)
# spurning4(plot=False)
# spurning5(plot=False)
# spurning6(plot=False)
# spurning7(plot=False)
# spurning8(plot=False)
spurning9(plot=True)
# spurning10(plot=False)
# spurning11(plot=False)
# spurning12(plot=True)
# frjals(plot=False)

og keyra það 
"""
