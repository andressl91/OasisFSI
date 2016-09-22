import numpy as np
import matplotlib.pyplot as plt


time = np.loadtxt('time.txt', delimiter=',')
Lift = np.loadtxt('Lift.txt', delimiter=',')
Drag = np.loadtxt('Drag.txt', delimiter=',')

plt.figure(1)
plt.plot(time, Lift)
plt.show()
