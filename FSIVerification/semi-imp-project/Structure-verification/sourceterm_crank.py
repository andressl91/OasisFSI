from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from structur_sympy import *
import sys
set_log_active(False)

sys.path.append('../Fluid_verification/tabulate')
from tabulate import tabulate

#default values
T = 1.0; dt = []; N = [];
solver = "chorin"; save_step = 1; fig = False; con = True

#command line arguments
while len(sys.argv) > 1:
    option = sys.argv[1]; del sys.argv[1];
    if option == "-T":
        T = float(sys.argv[1]); del sys.argv[1]
    elif option == "-dt":
        dt.append(float(sys.argv[1])); del sys.argv[1]
    elif option == "-N":
        N.append(int(sys.argv[1])); del sys.argv[1]
    elif option == "-L":
        L = float(sys.argv[1]); del sys.argv[1]
    elif option == "-save_step":
        save_step = float(sys.argv[1]); del sys.argv[1]
    elif option == "-fig":
        fig = bool(sys.argv[1]); del sys.argv[1]
    elif option == "-con":
        con = True;  del sys.argv[1]
    elif option == "-solver":
        solver = sys.argv[1]; del sys.argv[1]
    elif option.isdigit() == False:
        dt.append(float(option)); #del sys.argv[1]
    elif option.isdigit() == True:
        N.append(int(option));
    else:
        print sys.argv[0], ': invalid option', option

if len(N) == 0:
    N = [8]
if len(dt) == 0:
    dt = [0.1, 0.01]

def Solid(N, dt):
    mesh = UnitSquareMesh(N,N)
    V = VectorFunctionSpace(mesh, "CG", 1)

    d_exact, f = find_my_f_1()

    #PARAMETERS:
    rho_s = 1.0E3
    mu_s = 0.5E6
    nu_s = 0.4
    E_1 = 1.4E6
    lamda = nu_s*2.*mu_s/(1. - 2.*nu_s)

    t = dt
    k = Constant(dt)


    implementation = "1"

    #Second Piola Kirchhoff Stress tensor
    def s_s_n_l(d):
        I = Identity(2)
        F = I + grad(d)
        E = 0.5*((F.T*F) - I)
        #J = det(F)
        return (lamda*tr(E)*I + 2*mu_s*E)

    if implementation =="1":
        d_exact.t = 0
        f.t = 0; f_1 = f
        f.t = dt; f1 = f
        bcs = DirichletBC(V, d_exact, "on_boundary")
        psi = TestFunction(V)
        d = Function(V)
        d0 = Function(V)
        d1 = Function(V)
        d1 = interpolate(d_exact, V)
        d0 = interpolate(d_exact, V)

        G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx \
        + inner(0.5*(s_s_n_l(d) + s_s_n_l(d1)), grad(psi))*dx \
        - inner(0.5*(f1 + f_1), psi)*dx

    #Variational form with double spaces
    if implementation =="2":
        bc1 = DirichletBC(VV.sub(0), ((0,0)), boundaries, 1)
        bc2 = DirichletBC(VV.sub(1), ((0,0)), boundaries, 1)
        bcs = [bc1,bc2]
        psi, phi = TestFunctions(VV)
        wd = Function(VV)
        w, d = split(wd)
        w0d0 = Function(VV)
        w0, d0 = split(w0d0)

        G = rho_s/k*inner(w - w0, psi)*dx + inner(0.5*(s_s_n_l(d) + s_s_n_l(d0)), grad(psi))*dx \
        - inner(g,psi)*dx + dot(d - d0,phi)*dx - k*dot(0.5*(w + w0), phi)*dx

        #G = rho_s/k*inner(w - w0, psi)*dx + inner(s_s_n_l(0.5*(d + d0)), grad(psi))*dx \
        #- inner(g,psi)*dx + dot(d-d0,phi)*dx - k*dot(0.5*(w+w0),phi)*dx

    tic()
    while t < T:
        if implementation == "1":
            solve(G == 0,d, bcs, solver_parameters={"newton_solver": \
            {"relative_tolerance": 1E-7,"absolute_tolerance":1E-7,"maximum_iterations":100,"relaxation_parameter":1.0}})

            d1.assign(d0)
            d0.assign(d)
            d_exact.t = t
            f.t = t - dt; f_1 = f
            f.t = t + dt; f1 = f

            #dis_x.append(d(coord)[0])
            #dis_y.append(d(coord)[1])
            #time.append(t)
            if MPI.rank(mpi_comm_world()) == 0:
                print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

        if implementation == "2":
            solve(G == 0, wd, bcs, solver_parameters={"newton_solver": \
            {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
            w0d0.assign(wd)
            w,d = wd.split(True)

            #dis_x.append(d(coord)[0])
            #dis_y.append(d(coord)[1])
            #time.append(t)
            if MPI.rank(mpi_comm_world()) == 0:
                print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

        t += dt
    time.append(toc())
    d_exact.t = t
    #d_ = interpolate(d_exact,V2)

    #d_new = interpolate(d, V)
    L2 = errornorm(d_exact, d, norm_type = "l2", degree_rise = 2)
    print "HERERE", L2
    E.append(L2)
    h.append(mesh.hmin())

time = []; E = []; h = []
for t in dt:
    for n in N:
        Solid(n, t)

#for i in range(len(E)-1):
    #r = np.log(E[i+1]/E[i])/np.log(h[i+1]/h[i])
    #r = np.log(E[i+1]/E[i])/np.log(dt[i+1]/dt[i])
    #print "Here", r

if len(dt) == 1:
    check = N; opp = dt
else:
    check = dt; opp = N

if MPI.rank(mpi_comm_world()) == 0:
    print
    print "#################################### - L2 NORM - ####################################\n"
    table = []
    headers = ['N']
    for i in range(len(N)):
        li = []
        li.append(str(N[i]))
        for t in range(len(dt)):
            li.append("%e" % E[len(N)*t + i])
            li.append("%e" % time[len(N)*t + i])
        table.append(li)
    for i in range(len(dt)):
        headers.append("dt = %.g" % dt[i])
        headers.append("Runtime")
    print tabulate(table, headers)
    #print tabulate(table, headers, tablefmt="fancy_grid")


    if con == True:
        print
        print "############################### - CONVERGENCE RATE - ###############################\n"

        table = []
        headers = ['N']
        #for i in range(len(N)):
        for n in range(len(opp)):
            li = []
            li.append(str(opp[n]))
            for i in range(len(check)-1):
                #conv = np.log(E[i+1]/E[i])/np.log(check[i+1]/check[i])
                E_1 =  E[(i+1)*len(opp) + n]; E_0 =  E[i*len(opp) + n]
                conv = np.log(E_1/E_0)/np.log(check[i+1]/check[i])

                li.append(conv)
            table.append(li)
        for i in range(len(check)-1):
            headers.append("%g to %g" % (check[i], check[i+1]))
        print tabulate(table, headers)
        #print tabulate(table, headers, tablefmt="fancy_grid")
