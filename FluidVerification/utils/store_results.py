from dolfin import *
import numpy as np
import os

def store_results(case, rho, mu, common, Lift, Drag, Time, **namespace):

    if MPI.rank(mpi_comm_world()) == 0:

        vars().update(common)

        count = 1
        exist = False
        path = "./results/" + case + "/"
        while exist == False:
            if os.path.isdir(path + str(count)) == True:
                count += 1
            else:
                path += str(count)
                os.mkdir(path)
                exist = True

        Re = rho/mu

        f = open(path + "/parameters.txt", 'w')
        for i in range(len(common)):
            f.write('%s = %s \n' % (common.keys()[i], common[common.keys()[i]]))
        f.close()

        np.savetxt(path + '/dkdt.txt', Time, delimiter=',')
        np.savetxt(path + '/Lift.txt', Lift, delimiter=',')
        np.savetxt(path + '/Drag.txt', Drag, delimiter=',')
