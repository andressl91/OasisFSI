import numpy as np

def convergence(E_u, E_p, check):

    print "################# VELOCITY ################# \n"

    print "################# L2 - NORM #################\n"

    for i in E_u:
        print "errornorm", i

    print "################# Convergence Rate #################\n"

    for i in range(len(E_u)-1):
        r = np.log(E_u[i+1]/E_u[i]) / np.log(check[i+1]/check[i])
        print "Convergence rate", r

    print

    print "################# Pressure ################# \n"

    print "################# L2 - NORM #################\n"

    for i in E_p:
        print "errornorm", i

    print "################# Convergence Rate #################\n"

    for i in range(len(E_p)-1):
        r = np.log(E_p[i+1]/E_p[i]) / np.log(check[i+1]/check[i])
        print "Convergence rate", r

    print
    print
