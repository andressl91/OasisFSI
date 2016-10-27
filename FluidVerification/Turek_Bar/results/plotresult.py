#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt


time = np.loadtxt('./CFD3turek2/time.txt', delimiter=',')
Lift = np.loadtxt('./CFD3turek2/Lift.txt', delimiter=',')
Drag = np.loadtxt('./CFD3turek2/Drag.txt', delimiter=',')

time_ = np.loadtxt('./CFD3turek1_dt=0.001_vdeg=2/time.txt', delimiter=',')
Lift_ = np.loadtxt('./CFD3turek1_dt=0.001_vdeg=2
/Lift.txt', delimiter=',')
Drag_ = np.loadtxt('./CFD3turek1_dt=0.001_vdeg=2/Drag.txt', delimiter=',')


t_ = np.loadtxt('./CFD3turek2_dt=0.01_vdeg2/time.txt', delimiter=',')
L_v2_dt01 = np.loadtxt('./CFD3turek2_dt=0.01_vdeg2/Lift.txt', delimiter=',')
D_v2_dt01 = np.loadtxt('./CFD3turek2_dt=0.01_vdeg2/Drag.txt', delimiter=',')

#t_ = np.loadtxt('./CFD3turek1/time.txt', delimiter=',')
L_v1_dt01= np.loadtxt('./CFD3turek2_dt=0.01_vdeg1/Lift.txt', delimiter=',')
D_v1_dt01 = np.loadtxt('./CFD3turek2_dt=0.01_vdeg1/Drag.txt', delimiter=',')




plt.figure(1)

plt.subplot(222)
plt.title("CASE CFD3 dt = 0.001 T = 20 \n DOF = 306580,  cells = 75844  P2-P1 \n Liftforce ")
plt.axis([15, 16, -430, 400])
plt.plot(time, Lift )

#plt.figure(2)
plt.subplot(224)
plt.axis([15, 16, 434, 445])
plt.title("Dragforce ")
plt.plot(time, Drag)

#plt.figure(2)
plt.subplot(221)
plt.title("Case CFD3 dt = 0.001 T = 20 \n DOF = 28624,  cells = 6616  P2-P1 \n Liftforce")
plt.axis([15, 16, -400, 400])
plt.plot(time_, Lift_ )


plt.subplot(223)
plt.axis([15, 16, 429, 438])
plt.title("Dragforce ")
plt.plot(time_, Drag_)


plt.figure(2)

plt.subplot(222)
plt.title("CASE CFD3 dt = 0.01 T = 20 \n DOF = 77446,  cells = 75844 P1-P1 \n Liftforce ")
#plt.axis([15, 16, -430, 400])
plt.plot(t_, L_v1_dt01 )

plt.subplot(224)
#plt.axis([15, 16, 434, 445])
plt.title("Dragforce ")
plt.plot(t_, D_v1_dt01)

#plt.figure(2)
plt.subplot(221)
plt.title("Case CFD3 dt = 0.01 T = 20 \n DOF 306580,  cells = 75844 P2-P1 \n Liftforce")
#plt.axis([15, 16, -400, 400])
plt.plot(t_, L_v2_dt01 )


plt.subplot(223)
#plt.axis([15, 16, 429, 438])
plt.title("Dragforce ")
plt.plot(t_, D_v2_dt01)


plt.show()
