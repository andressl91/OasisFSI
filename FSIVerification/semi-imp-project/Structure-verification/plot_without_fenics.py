from os import path, listdir
from matplotlib.pyplot import *
import numpy as np

base = "Results/mixedspace"
types = listdir(base)

counter = 1
for t in types:
    #if t  "explicit": continue
    if t not in ["Implicit", "center", "reference"]: continue
    dts = listdir(path.join(base, t))
    for dt in dts:
        d_x = np.load(path.join(base, t, dt, "dis_x.np"))
        d_y = np.load(path.join(base, t, dt, "dis_y.np"))
        time = np.load(path.join(base, t, dt, "time.np"))

        plot(time, d_x, label=r"$\Delta t = %1.0e$, %s" % (float(dt), t))
        counter += 1
        hold("on")

xlim([0, 10])
ylim([-0.05, 0.02])
ncol = counter if counter < 4 else 3
legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
       ncol=ncol, fancybox=True, shadow=True, fontsize=15)
#plot([0, 10], [0, 0], color="k")
#plot([0, 10], [-0.135, -0.135], color="k")
show()
