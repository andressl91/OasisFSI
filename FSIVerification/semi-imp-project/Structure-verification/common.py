from os import path, makedirs
import numpy as np
import sys
import cPickle
import matplotlib.pyplot as plt

rel_path = path.dirname(path.abspath(__file__))

def solver_parameters(common, d):
    tmp = common.copy()
    tmp.update(d)

    # Update name of implementation and case path
    name = "Implicit" if tmp["E"] is None else tmp["E"].__name__
    case_path = path.join(rel_path, "results", tmp["space"], name, str(tmp["dt"]))
    tmp["name"] = name
    tmp["case_path"] = case_path
    return tmp


def viz(results, runs):
    # TODO: Get minimum time of all to plot over consistent
    plt.figure()
    for i, r in enumerate(results):
        displacement_x = np.array(r[0])
        displacement_y = np.array(r[1])
        time = np.array(r[2])
        simulation_parameters = runs[i]
        E = simulation_parameters["E"]
        dt = simulation_parameters["dt"]
        name = simulation_parameters["name"]
        case_path = simulation_parameters["case_path"]

        plt.plot(time, displacement_y, label=name)
        plt.ylabel("Displacement y")
        plt.xlabel("Time")
        plt.legend(loc=3)
        plt.hold("on")

        plt.title("implementation: %s, dt = %g, y_displacement" % (name, dt))

        # TODO: Move this to solver and use fenicsprobe
        if MPI.rank(mpi_comm_world()) == 0:
            if not path.exists(case_path):
                makedirs(case_path)

            print case_path
            displacement_x.dump(path.join(case_path, "dis_x.np"))
            displacement_t.dump(path.join(case_path, "dis_y.np"))
            time.dump(path.join(case_path, "time.np"))

            # Store simulation parameters
            f = open(path.join(case_path, "param.dat", 'w'))
            cPickle.dump(simulation_parameters, f)
            f.close()

    # FIXME: store in a consistent manner
    plt.savefig("test.png")
    plt.show()


