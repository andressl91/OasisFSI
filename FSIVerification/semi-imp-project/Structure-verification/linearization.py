from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from os import path, makedirs
set_log_active(False)

# First Piola Kirchoff stress tensor
def Piola1(d_, w_, E_func=None):
    I = Identity(2)
    if callable(E_func):
        E = E_func(d_, w_)
    else:
        F = I + grad(d_["n"])
        E = 0.5*((F.T*F) - I)

    return F*(lamda*tr(E)*I + 2*mu_s*E)

#Second Piola Kirchhoff Stress tensor
def Piola2(d_, w_, k, E_func=None):
    I = Identity(2)
    if callable(E_func):
        E = E_func(d_, w_, k)
    else:
        F = I + grad(d_["n"])
        E = 0.5*((F.T*F) - I)

    return lamda*tr(E)*I + 2*mu_s*E


def reference(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T \
             + grad(d_["n"]) + grad(d_["n-1"])) \
        + 0.5*(grad(d_["n"]).T * grad(d_["n"])) \
        + 0.5*(grad(d_["n-1"]).T * grad(d_["n-1"]))

    return 0.5*E


def naive_linearization(d_, w_, k):
    E = grad(d_["n"]) + grad(d_["n"]).T \
        + grad(d_["n-1"]).T*grad(d_["n"])

    return 0.5*E


def explicit(d_, w_, k):
    E = grad(d_["n-1"]) + grad(d_["n-1"]).T \
        + grad(d_["n-1"]).T*grad(d_["n-1"])

    return 0.5*E


def naive_ab(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T \
            + grad(d_["n"]) + grad(d_["n-1"])) \
        + 3./2 * (grad(d_["n-1"]).T*grad(d_["n-1"])) \
        - 1./2 * (grad(d_["n-1"]).T*grad(d_["n-2"]))

    return 0.5*E


def ab_before_cn(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T \
        + grad(d_["n"]) + grad(d_["n-1"])) \
        + (3/2.*grad(d_["n-1"]).T - 0.5*grad(d_["n-2"]).T)\
        * 0.5*(grad(d_["n"]) + grad(d_["n-1"]))

    return 0.5*E


def ab_before_cn_higher_order(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T \
        + grad(d_["n"]) + grad(d_["n-1"])) \
        + (23/12.*grad(d_["n-1"]).T - 4./3*grad(d_["n-2"]).T + 5/12.*grad(d_["n-3"]).T) \
        * 0.5*(grad(d_["n"]) + grad(d_["n-1"]))

    return 0.5*E


def cn_before_ab(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T + grad(d_["n"]) + grad(d_["n-1"]))  \
        + 0.5*((grad(d_["n-1"] + k*(3./2*w_["n-1"] - 1./2*w_["n-2"])).T * grad(d_["n"]).T) \
                + (grad(d_["n-1"]).T*grad(d_["n-1"])))

    return 0.5*E


def cn_before_ab_higher_order(d_, w_, k):
    E = 0.5*(grad(d_["n"]).T + grad(d_["n-1"]).T + grad(d_["n"]) + grad(d_["n-1"]))  \
        + 0.5*((grad(d_["n-1"] + k*(23./12*w_["n-1"] - 4./3*w_["n-2"] + 5/12.*w_["n-3"])).T \
                * grad(d_["n"]).T) + (grad(d_["n-1"]).T*grad(d_["n-1"])))

    return 0.5*E


# FIXME: Merge the two solver functions as only solver call is different
def solver_linear(G, d_, w_, wd_, bcs, T, dt):
    dis_x = []
    dis_y = []
    time = []

    a = lhs(G); L = rhs(G)
    while t <= T:
        # FIXME: Change to assemble seperatly and try different solver methods
        # (convex problem as long as we do not have bulking)
        solve(a == L, wd_["n"], bcs)

        # Update solution
        wd_["n-3"].vector().zero()
        wd_["n-3"].vector().axpy(1, wd_["n-2"].vector())
        wd_["n-2"].vector().zero()
        wd_["n-2"].vector().axpy(1, wd_["n-1"].vector())
        wd_["n-1"].vector().zero()
        wd_["n-1"].vector().axpy(1, wd_["n"].vector())
        # FIXME: Do you need to split again?
        #w0, d0 = wd0.split(True)

        # Get displacement
        dis_x.append(d_["n"](coord)[0])
        dis_y.append(d_["n"](coord)[1])
        time.append(t)

        t += dt
        if MPI.rank(mpi_comm_world()) == 0:
            print "Time: ",t

    return dis_x, dis_y, time


def solver_nonlinear(G, d_, w_, wd_, bcs, T, dt):
    dis_x = []; dis_y = []; time = []
    solver_parameters = {"newton_solver": \
                          {"relative_tolerance": 1E-8,
                           "absolute_tolerance": 1E-8,
                           "maximum_iterations": 100,
                           "relaxation_parameter": 1.0}}
    t = 0
    while t <= T:
        solve(G == 0, wd_["n"], bcs, solver_parameters=solver_parameters)

        # Update solution
        wd_["n-3"].vector().zero()
        wd_["n-3"].vector().axpy(1, wd_["n-2"].vector())
        wd_["n-2"].vector().zero()
        wd_["n-2"].vector().axpy(1, wd_["n-1"].vector())
        wd_["n-1"].vector().zero()
        wd_["n-1"].vector().axpy(1, wd_["n"].vector())
        # Do you need to split the functions again?
        #w0, d0 = wd0.split(True)
        w, d = wd_["n"].split(True)

        # Get displacement
	#from IPython import embed; embed()
        dis_x.append(d(coord)[0])
        dis_y.append(d(coord)[1])
        time.append(t)

        t += dt
        if MPI.rank(mpi_comm_world()) == 0:
            print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

    return dis_x, dis_y, time


def problem_mix(T, dt, E, coupling):
    # Temporal parameters
    t = 0
    k = Constant(dt)

    # Split problem to two 1.order differential equations
    psi, phi = TestFunctions(VV)

    # BCs
    bc1 = DirichletBC(VV.sub(0), ((0, 0)), boundaries, 1)
    bc2 = DirichletBC(VV.sub(1), ((0, 0)), boundaries, 1)
    bcs = [bc1, bc2]

    # Functions, wd is for holding the solution
    d_ = {}; w_ = {}; wd_ = {}
    for time in ["n", "n-1", "n-2", "n-3"]:
        if time == "n" and E not in [None, reference]:
            tmp_wd = Function(VV)
            wd_[time] = tmp_wd
            wd = TrialFunction(VV)
            w, d = split(wd)
        else:
            wd = Function(VV)
            wd_[time] = wd
            w, d = split(wd)

        d_[time] = d
        w_[time] = w

    # Time derivative
    if coupling == "center":
        G = rho_s/(2*k) * inner(w_["n"] - w_["n-2"], psi)*dx
    else:
        G = rho_s/k * inner(w_["n"] - w_["n-1"], psi)*dx

    # Stress tensor
    G += inner(Piola2(d_, w_, k, E_func=E), grad(psi))*dx

    # d-w coupling
    if coupling == "CN":
        G += inner(d_["n"] - d_["n-1"] - k*0.5*(w_["n"] - w_["n-1"]), phi)*dx
    elif coupling == "imp":
        G += inner(d_["n"] - d_["n-1"] - k*w_["n"], phi)*dx
    elif coupling == "exp":
        G += inner(d_["n"] - d_["n-1"] - k*w_["n-1"], phi)*dx
    elif coupling == "center":
        G += innter(d_["n"] - d_["n-2"] - 2*k*w["n-1"], phi)*dx
    else:
        print "The coupling %s is not implemented, 'CN', 'imp', and 'exp' are the only valid choices."
        sys.exit(0)

    # Solve
    if E in [None, reference]:
        displacement_x, displacement_y, time = solver_nonlinear(G, d_, w_, wd_, bcs, T, dt)
    else:
        displacement_x, displacement_y, time = solver_linear(G, d_, w_, wd_, bcs, T, dt)

    return displacement_x, displacement_y, time

def viz(results, runs):
    plt.figure()
    for i, r in enumerate(results):
        displacement_x = r[0]
        displacement_y = r[1]
        time = r[2]
        simulation_parameters = runs[i]
        E = simulation_parameters["E"]
        dt = simulation_parameters["dt"]
        space = simulation_parameters["space"]


        # TODO: Show implementation and dt
        # TODO: Read in a reference solution for comparison
        name = "None" if E is None else E.__name__
        plt.plot(time, displacement_y, label=name)
        plt.ylabel("Displacement y")
        plt.xlabel("Time")
        plt.legend(loc=3)

        plt.title("implementation: %s, dt = %g, y_displacement" % (name, dt))
        plt.savefig("Ydef.png")

        if MPI.rank(mpi_comm_world()) == 0:
            rel_path = path.dirname(path.abspath(__file__))
            case_path = path.join(rel_path, "results", space, E.__name__, str(dt))
            print case_path
            if not path.exists(case_path):
                makedirs(case_path)

            np.savetxt(path.join(case_path, "time.txt"), time, delimiter=",")
            np.savetxt(path.join(case_path, "dis_y.txt"), displacement_y, delimiter=',')

        # Name of text file coerced with +.txt
        #name = "./results/" + space + "/" + implementation + "/"+str(dt) + "/report.txt"
        # TODO: Store simulation parameters
        #f = open(name, 'w')
        #f.write("""Case parameters parameters\n """)
        #f.write("""T = %(T)g\ndt = %(dt)g\nImplementation = %(implementation)s""" %vars())
        #f.close()

    # FIXME: store in a consistent manner
    plt.show()


def solver_parameters(common, d):
    tmp = common.copy()
    tmp.update(d)
    return tmp


if __name__ == "__main__":
    # Set ut problem
    mesh = Mesh("von_karman_street_FSI_structure.xml")

    # Get the point [0.2,0.6] at the end of bar
    for coord in mesh.coordinates():
        if coord[0]==0.6 and (0.2 - DOLFIN_EPS <= coord[1] <= 0.2 + DOLFIN_EPS):
            print coord
            break

    V = VectorFunctionSpace(mesh, "CG", 2)
    VV=V*V

    BarLeftSide = AutoSubDomain(lambda x: "on_boundary" and \
                                (((x[0] - 0.2) * (x[0] - 0.2) +
                                 (x[1] - 0.2) * (x[1] - 0.2) < 0.0505*0.0505 )
                                 and x[1] >= 0.19 \
                                 and x[1] <= 0.21 \
                                 and x[0] > 0.2)
                               )

    boundaries = FacetFunction("size_t",mesh)
    boundaries.set_all(0)
    BarLeftSide.mark(boundaries,1)

    # Parameters:
    rho_s = 1.0E3
    mu_s = 0.5E6
    nu_s = 0.4
    E_1 = 1.4E6
    lamda = nu_s*2.*mu_s/(1. - 2.*nu_s)
    g = Constant((0,-2.*rho_s))
    beta = Constant(0.25)

    # Set up different numerical schemes
    common = {"space": "mixedspace",
              "E": None,         # Full implicte, not energy conservative
              "T": 0.3,          # End time
              "dt": 0.001,       # Time step
              "coupling": "CN", # Coupling between d and w
              "init": False      # Solve "exact" three first timesteps
             }

    # Nonlinear "reference" simulations
    ref = solver_parameters(common, {"E": reference})
    imp = solver_parameters(common, {"coupling": "imp"})

    # Linear, but not linearized
    exp = solver_parameters(common, {"E": explicit, "coupling": "exp"})
    center = solver_parameters(common, {"E": explicit, "coupling": "center"})

    # Linearization
    naive_lin = solver_parameters(common, {"E": naive_linearization})
    naive_ab = solver_parameters(common, {"E": naive_ab})
    ab_before_cn = solver_parameters(common, {"E": ab_before_cn})
    ab_before_cn_higher_order = solver_parameters(common, {"E": ab_before_cn_higher_order})
    cn_before_ab = solver_parameters(common, {"E": cn_before_ab})
    cn_before_ab_higher_order = solver_parameters(common, {"E": cn_before_ab_higher_order})

    # Set-ups to run
    runs = [ref] #,
            #imp,
            #exp,
            #naive_lin,
            #naive_ab,
            #ab_before_cn,
            #ab_before_cn_higher_order,
            #cn_before_ab,
            #cn_before_ab_higher_order]
    results = []
    for r in runs:
        if r["space"] == "mixedspace":
            tmp_res = problem_mix(r["T"], r["dt"], r["E"], r["coupling"])
            results.append(tmp_res)
        elif r["space"] == "singlespace":
            # FIXME: Not implemented 
            problem_lin()
        else:
            print "Problem type %s is not implemented, only mixedspace " \
                   + "and singlespace are valid options" % r["space"]
            sys.exit(0)

    viz(results, runs)
