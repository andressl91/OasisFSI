from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys, os
set_log_active(False)

mesh = Mesh("von_karman_street_FSI_structure.xml")

for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break

V = VectorFunctionSpace(mesh, "CG", 2)
VV=V*V
print "Dofs: ",V.dim(), "Cells:", mesh.num_cells()

BarLeftSide =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
BarLeftSide.mark(boundaries,1)
#plot(boundaries,interactive=True)

#PARAMETERS:
rho_s = 1.0E3
mu_s = 0.5E6
nu_s = 0.4
E_1 = 1.4E6
lamda = nu_s*2.*mu_s/(1. - 2.*nu_s)
g = Constant((0,-2.*rho_s))

beta = Constant(0.25)

list_krylov_solver_methods()
list_krylov_solver_preconditioners()

#Second Piola Kirchhoff Stress tensor
def Piola2(d, lamda, mu):
    I = Identity(2)
    F = I + grad(d)
    E = 0.5*((F.T*F) - I)

    return (lamda*tr(E)*I + 2*mu_s*E)

def simple_lin_2(d_n, d_n0, d_n1, lamda, mu):
    I = Identity(2)
    E = 1./2 * (grad(d_n) + grad(d_n).T + (3./2*grad(d_n0).T  - 1./2*grad(d_n1).T) *grad(d_n))

    return lamda*tr(E)*I + 2*mu_s*E


def simple_lin(d_n, d_n1, lamda, mu):
    I = Identity(2)
    E = 1./2 * (grad(d_n) + grad(d_n).T + grad(d_n1).T*grad(d_n))

    return lamda*tr(E)*I + 2*mu_s*E

def simple_lin_crank(d_n, d_n0, d_n1, lamda, mu):
    I = Identity(2)
    E = 1./2* ( 0.5*(grad(d_n).T + grad(d_n0).T + grad(d_n) + grad(d_n0)) +  grad(d_n1).T*grad(d_n))

    return lamda*tr(E)*I + 2*mu_s*E


def piola_adam_single(d_n, d_n1, d_n2, lamda, mu):
    I = Identity(2)
    E = 1./2* ( 0.5*(grad(d_n).T + grad(d_n2).T + grad(d_n) + grad(d_n2)) \
    + (3./2*grad(d_n1).T - 1./2*grad(d_n2).T) + 1./2*(grad(d_n) + grad(d_n2)))
    return (lamda*tr(E)*I + 2*mu_s*E)

"""

#Why does this work when it is fully imp, LHS is centered at n-1
def piola_adam_single(d_n, d_n1, d_n2, lamda, mu):
    I = Identity(2)
    E = 1./2* ( grad(d_n).T  + grad(d_n)  \
    + (3./2*grad(d_n1).T - 1./2*grad(d_n2).T)* grad(d_n)  )

    return (lamda*tr(E)*I + 2*mu_s*E)
"""
#REMEMBER TO TURN OF STORAGE WHEN TESTING!!!

def piola2_adam_double(d_n, d_n1, d_n2, lamda, mu):
    I = Identity(2)
    E = 1./2* ( 0.5*(grad(d_n).T + grad(d_n1).T + grad(d_n) + grad(d_n1)) \
    +  (3./2*grad(d_n1).T - 1./2*grad(d_n2).T)* 0.5*(grad(d_n).T + grad(d_n1).T) )

    return (lamda*tr(E)*I + 2*mu_s*E)
"""
def piola2_adam_double(d_n, d_n1, d_n2, lamda, mu):
    I = Identity(2)
    E = 1./2* ( grad(d_n).T + grad(d_n)  + (3./2*grad(d_n1).T - 1./2*grad(d_n2).T)*grad(d_n) )
    return (lamda*tr(E)*I + 2*mu_s*E)
"""

def solver(T, dt, space, implementation, betterstart):

    t = 0
    k = Constant(dt)
    dis_x = []; dis_y = []; time = []
    plt.figure(1)

    if space == "singlespace":
            bcs = [DirichletBC(V, ((0, 0)), boundaries, 1)]
            psi = TestFunction(V)
            d0 = Function(V)
            d1 = Function(V)
            if betterstart == True:
                d = Function(V)
                G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                + 0.5*inner(Piola2(d, lamda, mu_s) + Piola2(d1, lamda, mu_s), grad(psi))*dx

                while t < 2*dt:
                    solve(G == 0, d, bcs, solver_parameters={"newton_solver": \
                    {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
                    d1.assign(d0)
                    d0.assign(d)

                    dis_x.append(d0(coord)[0])
                    dis_y.append(d0(coord)[1])
                    time.append(t)
                    if MPI.rank(mpi_comm_world()) == 0:
                        print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

                    t += dt

            if implementation == "C-N":
                d = Function(V)
                G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                + 0.5*inner(Piola2(d, lamda, mu_s) + Piola2(d1, lamda, mu_s), grad(psi))*dx

            if implementation == "Piccard":
                # Define variational problem for Picard iteration
                d_k = Function(V)
                d_k.assign(d0)
                d = TrialFunction(V)
                G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                + inner(simple_lin(d, d_k, lamda, mu_s), grad(psi))*dx

            if implementation == "Lin-CN":
                d = TrialFunction(V)
                #G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                #+ inner(simple_lin_crank(d, d0, d1, lamda, mu_s), grad(psi))*dx
                #ORG!!
                G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                + 0.5*inner(simple_lin(d, d0, lamda, mu_s) + Piola2(d1, lamda, mu_s), grad(psi))*dx

            if implementation == "simple_lin":
                d = TrialFunction(V)
                G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                + inner(simple_lin(d, d0, lamda, mu_s), grad(psi))*dx

            if implementation == "A-B":
                d = TrialFunction(V)
                G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                + inner(piola_adam_single(d, d0, d1, lamda, mu_s), grad(psi))*dx

            #Why does newtonit give different answer for simple_lin ??
            if implementation == "C-N":
                tic()
                while t <= T:
                    solve(G == 0, d, bcs, solver_parameters={"newton_solver": \
                    {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
                    d1.assign(d0)
                    d0.assign(d)

                    dis_x.append(d0(coord)[0])
                    dis_y.append(d0(coord)[1])
                    time.append(t)

                    t += dt
                    if MPI.rank(mpi_comm_world()) == 0:
                        print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]
                comp_time.append(toc())

            if implementation == "Piccard":
                tic()
                # Picard iterations
                a = lhs(G); L = rhs(G)
                #A = assemble(a); b = None
                d_sol = Function(V)
                maxiter = 10        # max no of iterations allowed
                tol = 1.0E-7        # tolerance

                while t <= T:
                    eps = 1.0           # error measure ||u-u_k||
                    iter = 0            # iteration counter
                    #b = assemble(L, tensor=b)
                    while eps > tol and iter < maxiter:
                        iter += 1
                        solve(a == L, d_sol, bcs)
                        #A = assemble(a)
                        #[bc.apply(A,b) for bc in bcs]
                        #solve(A, d_sol.vector(), b)
                        #diff = d_sol.vector().array() - d_k.vector().array()
                        #eps = np.linalg.norm(diff, ord=np.Inf)
                        eps = errornorm(d_sol, d_k,degree_rise=3)
                        print 'iter=%d: norm=%g' % (iter, eps)
                        d_k.assign(d_sol)   # update for next iteration


                    d1.assign(d0)
                    d0.assign(d_sol)

                    dis_x.append(d0(coord)[0])
                    dis_y.append(d0(coord)[1])
                    time.append(t)

                    t += dt
                    if MPI.rank(mpi_comm_world()) == 0:
                        print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]
                comp_time.append(toc())

            if implementation == "simple_lin" or "A-B" or "Lin-CN":
                a = lhs(G); L = rhs(G)
                #pc = PETScPreconditioner("jacobi")
                #sol = PETScKrylovSolver("default")
                #b = None
                d_sol = Function(V)
                tic()
                while t <= T:
                    #A = assemble(a)
                    #b = assemble(L, tensor=b)
                    #[bc.apply(A, b) for bc in bcs]
                    #solve(A, d_sol.vector(), b)
                    solve(lhs(G) == rhs(G), d_sol, bcs)
                    d1.assign(d0)
                    d0.assign(d_sol)

                    dis_x.append(d0(coord)[0])
                    dis_y.append(d0(coord)[1])
                    time.append(t)

                    t += dt
                    if MPI.rank(mpi_comm_world()) == 0:
                        print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

                comp_time.append(toc())
                plt.plot(time, dis_y, label = implementation)
                plt.ylabel("Displacement y")
                plt.xlabel("Time")
                plt.legend(loc=3)



    if space == "mixedspace":
    #Split problem to two 1.order differential equations
        psi, phi = TestFunctions(VV)
        bc1 = DirichletBC(VV.sub(0), ((0, 0)), boundaries, 1)
        bc2 = DirichletBC(VV.sub(1), ((0, 0)), boundaries, 1)
        bcs = [bc1, bc2]
        wd = Function(VV)
        w, d = split(wd)
        wd0 = Function(VV)
        w0, d0 = split(wd0)
        wd_1 = Function(VV)
        w_1, d_1 = split(wd_1)

        if betterstart == True:
            G = rho_s/k*inner(w - w0, psi)*dx + 0.5*inner(Piola2(d, lamda, mu_s) + Piola2(d0, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx + dot(d - d0,phi)*dx - k*dot(w, phi)*dx#- k*dot(0.5*(w + w0),phi)*dx

            while t < 2*dt:
                print "HERE"
                solve(G == 0, wd, bcs, solver_parameters={"newton_solver": \
                {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})

                wd_1.assign(wd0)
                wd0.assign(wd)
                w0, d0 = wd0.split(True)

                dis_x.append(d0(coord)[0])
                dis_y.append(d0(coord)[1])
                time.append(t)
                t += dt

                if MPI.rank(mpi_comm_world()) == 0:
                    print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

        if implementation =="C-N":
            G = rho_s/k*inner(w - w0, psi)*dx + 0.5*inner(Piola2(d, lamda, mu_s) + Piola2(d0, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx + dot(d - d0, phi)*dx - k*dot(0.5*(w + w0), phi)*dx

        if implementation == "Piccard":
            # Define variational problem for Picard iteration
            wd = TrialFunction(VV)
            w, d = split(wd)
            wd_k = Function(VV)
            w_k, d_k = wd_k.split(VV)
            #d_k.assign(d0)
            #CHECK WITCH IS BEST
            #G = rho_s/k*inner(w - w0, psi)*dx + inner(simple_lin(d, d_k, lamda, mu_s), grad(psi))*dx \
            #- inner(g, psi)*dx + dot(d - d0, phi)*dx - k*inner(w, phi)*dx #- k*dot(0.5*(w + w0), phi)*dx

            G = rho_s/k*inner(w - w0, psi)*dx + inner(simple_lin_crank(d, d0, d_k, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx + dot(d - d0, phi)*dx - k*dot(0.5*(w + w0), phi)*dx# k*inner(w, phi)*dx #


        if implementation == "Lin-CN":
            wd = TrialFunction(VV)
            w, d = split(wd)
            G = rho_s/k*inner(w - w0, psi)*dx + 0.5*inner(simple_lin_2(d, d0, d_1, lamda, mu_s) + Piola2(d0, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx + dot(d - d0, phi)*dx - k*dot(0.5*(w + w0), phi)*dx

        if implementation == "simple_lin":
            wd = TrialFunction(VV)
            w, d = split(wd)

            G = rho_s/k*inner(w - w0, psi)*dx + inner(simple_lin_2(d, d0, d_1, lamda, mu_s), grad(psi) )*dx \
            - inner(g, psi)*dx + 1./k*inner(d - d0, phi)*dx - 0.5*inner(w + w0, phi)*dx#- inner(w, phi)*dx


        if implementation =="A-B":
            wd = TrialFunction(VV)
            w, d = split(wd)
            G = rho_s/k*inner(w - w0, psi)*dx + inner(piola2_adam_double(d, d0, d_1, lamda, mu_s), grad(psi) )*dx \
            - inner(g, psi)*dx \
            + 1./k*inner(d - d0, phi)*dx - inner(w, phi)*dx#k*dot(0.5*(w + w0),phi)*dx# i#
            #+ 1./k*inner(d - d0, phi)*dx - inner(w, phi)*dx#


    #dis_file = File("results/x_direction.pvd")
        if implementation == "C-N":
            tic()
            while t <= T:
                solve(G == 0, wd, bcs, solver_parameters={"newton_solver": \
                {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})

                wd_1.assign(wd0)
                wd0.assign(wd)
                w0, d0 = wd0.split(True)
                w, d = wd.split(True)

                dis_x.append(d0(coord)[0])
                dis_y.append(d0(coord)[1])
                time.append(t)

                t += dt
                if MPI.rank(mpi_comm_world()) == 0:
                    print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]
            comp_time.append(toc())

        if implementation == "Piccard":
            tic()
            # Picard iterations
            a = lhs(G); L = rhs(G)
            #A = assemble(a);
            b = None
            wd_sol = Function(VV)
            maxiter = 1       # max no of iterations allowed
            tol = 1.0E-7        # tolerance

            while t <= T:
                eps = 1.0           # error measure ||u-u_k||
                iter = 0            # iteration counter
                b = assemble(L, tensor=b)
                while eps > tol and iter < maxiter:
                    iter += 1
                    #solve(a == L, wd_sol, bcs)
                    A = assemble(a)
                    [bc.apply(A,b) for bc in bcs]
                    solve(A, wd_sol.vector(), b)
                    #diff = d_sol.vector().array() - d_k.vector().array()
                    #eps = np.linalg.norm(diff, ord=np.Inf)
                    w_, d_ = wd_sol.split(True)
                    w_k, d_k = wd_k.split(True)
                    eps = errornorm(d_, d_k,degree_rise=3)
                    #eps = errornorm(wd_sol, wd_k,degree_rise=3)
                    print 'iter=%d: norm=%g' % (iter, eps)
                    wd_k.assign(wd_sol)   # update for next iteration


                wd_1.assign(wd0)
                wd0.assign(wd_sol)
                w0, d0 = wd0.split(True)

                dis_x.append(d0(coord)[0])
                dis_y.append(d0(coord)[1])
                time.append(t)

                t += dt
                if MPI.rank(mpi_comm_world()) == 0:
                    print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]
            comp_time.append(toc())


        if implementation == "simple_lin" or "A-B" or "Lin-CN":
            a = lhs(G); L = rhs(G)
            #A = assemble(a); b = None
            wd_sol = Function(VV)
            tic()
            while t <= T:
                #b = assemble(L, tensor=b)
                #[bc.apply(A, b) for bc in bcs]
                #solve(A, wd_sol.vector(), b)
                solve(a == L, wd_sol, bcs)
                wd_1.assign(wd0)
                wd0.assign(wd_sol)
                w0, d0 = wd0.split(True)

                dis_x.append(d0(coord)[0])
                dis_y.append(d0(coord)[1])
                time.append(t)

                t += dt
                if MPI.rank(mpi_comm_world()) == 0:
                    print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]
            comp_time.append(toc())


        plt.plot(time, dis_y, label = implementation)
        plt.ylabel("Displacement y")
        plt.xlabel("Time")
        plt.legend(loc=3)


    plt.title("dt = %g, y_dir" % (dt))
    plt.savefig("Ydef.png")


#    if MPI.rank(mpi_comm_world()) == 0:
#        if os.path.exists("./results/" + space + "/" + implementation + "/"+str(dt)) == False:
#           os.makedirs("./results/" + space + "/" + implementation + "/"+str(dt))
#
#        np.savetxt("./results/" + space + "/" + implementation + "/"+str(dt)+"/time.txt", time, delimiter=',')
#        np.savetxt("./results/" + space + "/" + implementation + "/"+str(dt)+"/dis_y.txt", dis_y, delimiter=',')
#
    #    name = "./results/" + space + "/" + implementation + "/"+str(dt) + "/report.txt"  # Name of text file coerced with +.txt
    #    f = open(name, 'w')
    #    f.write("""Case parameters parameters\n """)
    #    f.write("""T = %(T)g\ndt = %(dt)g\nImplementation = %(implementation)s
    #    """ %vars())
    #    f.close()

    #plt.show()

#space = ["singlespace"]
space = ["mixedspace"]
#implementation = ["Piccard" ]# "simple_lin", "C-N"]
implementation = ["A-B"]
#implementation = ["Lin-CN"]
#implementation = ["Lin-CN", "C-N", "A-B", "simple_lin"]

comp_time = []
runs = []
#AB1 = {"space": "mixedspace", "implementation": "A-B", "T": 0.5, "dt": 0.005, "betterstart": False}; runs.append(AB1)
#AB2 = {"space": "mixedspace", "implementation": "A-B", "T": 0.5, "dt": 0.005, "betterstart": True}; runs.append(AB2)
#AB3 = {"space": "singlespace", "implementation": "A-B", "T": 0.5, "dt": 0.005, "betterstart": False}; runs.append(AB3)
#AB4 = {"space": "singlespace", "implementation": "A-B", "T": 0.5, "dt": 0.005, "betterstart": True}; runs.append(AB4)

#PI1 = {"space": "singlespace", "implementation": "Piccard", "T": 0.2, "dt": 0.005, "betterstart": True}; runs.append(PI1)
#PI2 = {"space": "singlespace", "implementation": "Piccard", "T": 0.2, "dt": 0.005, "betterstart": False}; runs.append(PI2)
PI3 = {"space": "mixedspace", "implementation": "Piccard", "T": 0.5, "dt": 0.005, "betterstart": False}; runs.append(PI3)

#SI1 = {"space": "singlespace", "implementation": "simple_lin", "T": 0.3, "dt": 0.005, "betterstart": True}; runs.append(SI1)
#SI1 = {"space": "singlespace", "implementation": "simple_lin", "T": 0.3, "dt": 0.005, "betterstart": False}; runs.append(SI1)
SI2 = {"space": "mixedspace", "implementation": "simple_lin", "T": 0.5, "dt": 0.005, "betterstart": False}; runs.append(SI2)

#LC1 = {"space": "singlespace", "implementation": "Lin-CN", "T": 0.3, "dt": 0.005, "betterstart": True}; runs.append(LC1)
#LC1 = {"space": "singlespace", "implementation": "Lin-CN", "T": 0.3, "dt": 0.005, "betterstart": False}; runs.append(LC1)
#LC2 = {"space": "mixedspace", "implementation": "Lin-CN", "T": 0.5, "dt": 0.005, "betterstart": False}; runs.append(LC2)

#CN1 = {"space": "singlespace", "implementation": "C-N", "T": 0.5, "dt": 0.02, "betterstart": False}; runs.append(CN1)
CN2 = {"space": "mixedspace", "implementation": "C-N", "T": 0.5, "dt": 0.005, "betterstart": False}; runs.append(CN2)

for r in runs:
    print r["implementation"], r["betterstart"]
    solver(r["T"], r["dt"], r["space"], r["implementation"], r["betterstart"])

for i in range(len(runs)):
    print "%s -- > CPU TIME %f" % (runs[i]["implementation"], comp_time[i])

    #plt.show()

"""
T = 1.0
dt = float(sys.argv[1])

for s in space:
    for i in implementation:
        print "Solving for space" + s + "implementation" + i
        solver(T, dt, s, i, betterstart = False)
"""
#for i in range(len(space)):
#    for j in range(len(implementation)):
#        print "%s -- > CPU TIME %f" % (implementation[j], comp_time[i*len(implementation) + j])

    #plt.show()
"""
plt.figure(1)
plt.title("implementation %s, x-dir" % (implementation))
plt.plot(time,dis_x,);title; plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
plt.savefig("run_x_imp%s.jpg" % (implementation))
#plt.show()
"""
