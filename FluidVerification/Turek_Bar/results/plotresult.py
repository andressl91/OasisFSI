#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt


time = np.loadtxt('time.txt', delimiter=',')
Lift = np.loadtxt('Lift.txt', delimiter=',')
Drag = np.loadtxt('Drag.txt', delimiter=',')

time_ = np.loadtxt('./CFD3 turek1/time.txt', delimiter=',')
Lift_ = np.loadtxt('./CFD3 turek1/Lift.txt', delimiter=',')
Drag_ = np.loadtxt('./CFD3 turek1/Drag.txt', delimiter=',')


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


#plt.figure(1)

"""
f, axarr = plt.subplots(2, 2)
axarr[0, 0].plot(time, Lift)
axarr[0, 0].axis([9, 10, -410, 400] )
axarr[0, 0].set_title('Lift')
axarr[0, 1].plot(time_, Lift_)
axarr[0, 1].axis([9, 10, -400, 400] )
axarr[0, 1].set_title('Lift_')
axarr[1, 0].plot(time, Drag)
axarr[1, 0].axis([9, 10, 434, 445] )
axarr[1, 0].set_title('Drag')
axarr[1, 1].plot(time_, Drag_)
axarr[1, 1].axis([9, 10, 434, 440] )
axarr[1, 1].set_title('Drag_')
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
"""
plt.show()
