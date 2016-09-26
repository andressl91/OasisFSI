import numpy as np
import matplotlib.pyplot as plt


time = np.loadtxt('time.txt', delimiter=',')
Lift = np.loadtxt('Lift.txt', delimiter=',')
Drag = np.loadtxt('Drag.txt', delimiter=',')

plt.figure(1)
plt.subplot(221)
plt.title("Liftforce CFD3 Turek")
plt.axis([15, 16, -400, 400])
plt.plot(time, Lift )

#plt.figure(2)
plt.subplot(222)
plt.axis([15, 16, 428, 438])
plt.title("Dragforce CFD3 Turek")
plt.plot(time, Drag)


plt.show()
