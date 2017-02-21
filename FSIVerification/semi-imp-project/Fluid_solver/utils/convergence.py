import numpy as np
from tabulate import tabulate

def convergence(E_u, E_p, N, dt):

    check = N if len(N) >= len(dt) else dt
    opp = dt if check == N else N

    if check == N:
        print
        print "#################################### - ERROR/CON SPACE - ####################################\n"

    else:
        print
        print "#################################### - ERROR/CON TIME - ####################################\n"

    time = [i for i in range(len(E_u))]
    for E in [E_u]:
        print
        print "#################################### - L2 NORM - ####################################\n"
        table = []
        headers = ["N" if opp is N else "dt"]
        #headers = ["N"]
        for i in range(len(opp)):
            li = []
            li.append(str(opp[i]))
            for t in range(len(check)):
                li.append("%e" % E[i*len(check) + t])
                li.append("%e" % time[i*len(check) + t]) #SJEKKKKK!!
            table.append(li)
        for i in range(len(check)):
            headers.append("dt = %.g" % check[i] if check is dt else "N = %g" % check[i])
            headers.append("Runtime")
        print tabulate.tabulate(table, headers, tablefmt="fancy_grid")

        print


        print
        print "############################### - CONVERGENCE RATE - ###############################\n"

        table = []
        headers = ["N" if opp is N else "dt"]
        #for i in range(len(N)):

        for n in range(len(opp)):
            li = []
            li.append(str(opp[n]))
            for i in range(len(check)-1):
                #conv = np.log(E[i+1]/E[i])/np.log(check[i+1]/check[i])
                error = E[n*len(check) + (i+1)] / E[n*len(check) + i]
                h_ = check[n*len(check) + (i+1)] / check[n*len(check) + i]
                conv = np.log(error)/np.log(h_) #h is determined in main solve method

                li.append(conv)
            table.append(li)
        for i in range(len(check)-1):
            headers.append("%g to %g" % (check[i], check[i+1]))
        print tabulate.tabulate(table, headers, tablefmt="fancy_grid")
        #time = []; E = []; h = []
