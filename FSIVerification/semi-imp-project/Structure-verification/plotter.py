import numpy as np
import matplotlib.pyplot as plt
import os

#space = "mixedspace"
#implementation = "simple_lin"
#dt = 0.02

#time = np.loadtxt("./results/" + space + "/" + implementation + "/"+str(dt)+"/time.txt", delimiter=',')
#dis_y = np.loadtxt("./results/" + space + "/" + implementation + "/"+str(dt)+"/dis_y.txt", delimiter=',')
t_02 = np.loadtxt("./results/singlespace/A-B/0.02/time.txt", delimiter=',')
t_002 = np.loadtxt("./results/singlespace/A-B/0.002/time.txt", delimiter=',')
t_0005 = np.loadtxt("./results/mixedspace/simple_lin/0.0005/time.txt", delimiter=',')

#Crank-Nic implementations SINGLESPACE
CN_02_S = np.loadtxt("./results/singlespace/C-N/0.02/dis_y.txt", delimiter=',')
CN_002_S = np.loadtxt("./results/singlespace/C-N/0.002/dis_y.txt", delimiter=',')
#CN_0005_S = np.loadtxt("./results/singlespace/C-N/0.0005/dis_y.txt", delimiter=',')

#Crank-Nic implementations MIXEDSPACE
CN_02_D = np.loadtxt("./results/mixedspace/C-N/0.02/dis_y.txt", delimiter=',')
CN_002_D = np.loadtxt("./results/mixedspace/C-N/0.002/dis_y.txt", delimiter=',')
#CN_0005_D = np.loadtxt("./results/mixedspace/C-N/0.0005/dis_y.txt", delimiter=',')

#Adam-Bashford implementations SINGLESPACE
AB_02_S = np.loadtxt("./results/singlespace/A-B/0.02/dis_y.txt", delimiter=',')
AB_002_S = np.loadtxt("./results/singlespace/A-B/0.002/dis_y.txt", delimiter=',')
AB_0005_S = np.loadtxt("./results/singlespace/A-B/0.0005/dis_y.txt", delimiter=',')

#Adam-Bashford implementations MIXEDSPACE
AB_02_D = np.loadtxt("./results/mixedspace/A-B/0.02/dis_y.txt", delimiter=',')
AB_002_D = np.loadtxt("./results/mixedspace/A-B/0.002/dis_y.txt", delimiter=',')
AB_0005_D = np.loadtxt("./results/mixedspace/A-B/0.0005/dis_y.txt", delimiter=',')

#Simple linearisation implementations SINGLESPACE
SL_02_S = np.loadtxt("./results/singlespace/simple_lin/0.02/dis_y.txt", delimiter=',')
SL_002_S = np.loadtxt("./results/singlespace/simple_lin/0.002/dis_y.txt", delimiter=',')
SL_0005_S = np.loadtxt("./results/singlespace/simple_lin/0.0005/dis_y.txt", delimiter=',')

#Simple linearisation implementations mixedspace
SL_02_D = np.loadtxt("./results/mixedspace/simple_lin/0.02/dis_y.txt", delimiter=',')
SL_002_D = np.loadtxt("./results/mixedspace/simple_lin/0.002/dis_y.txt", delimiter=',')
SL_0005_D = np.loadtxt("./results/mixedspace/simple_lin/0.0005/dis_y.txt", delimiter=',')

plt.figure(1)
#Crank-Nic

#plt.plot(t_02, CN_02_S, label="CN_02_S")
#plt.plot(t_002, CN_002_S, label="CN_002_S")
#plt.plot(t_0005, CN_0005_S, label="CN_0005_S")

#plt.plot(t_02, CN_02_D, label="CN_02_D")
plt.plot(t_002, CN_002_D, label="CN_002_D")
#plt.plot(t_0005, CN_0005_D, label="CN_0005_D")

#Adam-Bashford
#plt.plot(t_02, AB_02_S, label="AB_02_S")
#plt.plot(t_002, AB_002_S, label="AB_002_S")
plt.plot(t_0005, AB_0005_S, label="AB_0005_S")

#plt.plot(t_02, AB_02_D, label="AB_02_D")
#plt.plot(t_002, AB_002_D, label="AB_002_D")
#plt.plot(t_0005, AB_0005_D, label="AB_0005_D")

#Simple linearisation
#plt.plot(t_02, SL_02_S, label="SL_02_S")
#plt.plot(t_002, SL_002_S, label="SL_002_S")
plt.plot(t_0005, SL_0005_S, label="SL_0005_S")

#plt.plot(t_02, SL_02_D, label="SL_02_D")
#plt.plot(t_002, SL_002_D, label="SL_002_D")
#plt.plot(t_0005, SL_0005_D, label="SL_0005_D")
plt.legend(loc=3)

plt.show()

#plt.fgure(1)
#plt.plot(time, dis_y)
#plt.show()


"""
single_data = []
for root, dirs, files in os.walk("/Users/Andreas/Desktop/OasisFSI/FSIVerification/semi-imp-project/Structure-verification/results/singlespace/"):
    path = root
    A = root.split("/")
    print A
    single_data.append(A[-1])
    #try:
#            float(A[-1])
#            print "s"
    #except ValueError:
    #        print "sda"

"""
