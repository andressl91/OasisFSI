import sys, os
import matplotlib.pyplot as plt
import numpy as np
from argpar import parse



def postpro(Lift, Drag, dis_x, dis_y, time, Re, m, U_dof, run_time, mesh_cells, case):
    args = parse()
    T = args.T; dt = args.dt
    v_deg = args.v_deg; p_deg = args.p_deg
    d_deg = args.d_deg
    theta = args.theta
    fig = False
    count = 1

    count = 1
    while os.path.exists("./experiments/fsi1/"+str(count)):
        count+= 1

    os.makedirs("./experiments/fsi1/"+str(count))

    print("Creating report file ./experiments/fsi1/"+str(count)+"/report.txt")
    name = "./experiments/fsi1/"+str(count)+"/report.txt"  # Name of text file coerced with +.txt
    f = open(name, 'w')
    f.write("FSI1 Turek parameters\n"
            "Re = %(Re)g \nmesh = %(m)s\nDOF = %(U_dof)d\nT = %(T)g\ndt = %(dt)g\nv_deg = %(v_deg)g\n,d_deg%(d_deg)g\np_deg = %(p_deg)g\n"
            "theta_scheme = %(theta).1f\n" % vars())
    f.write("Runtime = %f \n\n" % run_time)

    f.write("Steady Forces:\nLift Force = %g\n"
            "Drag Force = %g\n\n" % (Lift[-1], Drag[-1]))
    f.write("Steady Displacement:\ndisplacement_x = %g \n"
    "displacement_y = %g \n" % (dis_x[-1], dis_y[-1]))
    f.close()

    np.savetxt("./experiments/fsi1/"+str(count)+"/Lift.txt", Lift, delimiter=',')
    np.savetxt("./experiments/fsi1/"+str(count)+"/Drag.txt", Drag, delimiter=',')
    np.savetxt("./experiments/fsi1/"+str(count)+"/time.txt", time, delimiter=',')

    plt.figure(1)
    plt.title("LIFT \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
    plt.xlabel("Time Seconds")
    plt.ylabel("Lift force Newton")
    plt.plot(time, Lift, label='dt  %g' % dt)
    plt.legend(loc=4)
    plt.savefig("./experiments/fsi1/"+str(count)+"/lift.png")

    plt.figure(2)
    plt.title("DRAG \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
    plt.xlabel("Time Seconds")
    plt.ylabel("Drag force Newton")
    plt.plot(time, Drag, label='dt  %g' % dt)
    plt.legend(loc=4)
    plt.savefig("./experiments/fsi1/"+str(count)+"/drag.png")
    #plt.show()
    plt.figure(3)
    plt.title("Dis_x \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
    plt.xlabel("Time Seconds")
    plt.ylabel("Drag force Newton")
    plt.plot(time, dis_x, label='dt  %g' % dt)
    plt.legend(loc=4)
    plt.savefig("./experiments/fsi1/"+str(count)+"/dis_x.png")

    plt.figure(4)
    plt.title("Dis_y \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
    plt.xlabel("Time Seconds")
    plt.ylabel("Drag force Newton")
    plt.plot(time, dis_y, label='dt  %g' % dt)
    plt.legend(loc=4)
    plt.savefig("./experiments/fsi1/"+str(count)+"/dis_y.png")


    #vel_file << u
    print "Discretization theta = %g" % theta
    print "Lift %g" % Lift[-1]
    print "Drag %g" % Drag[-1]
    print "Displacement x %g" % dis_x[-1]
    print "displacement_y %g" % dis_y[-1]
