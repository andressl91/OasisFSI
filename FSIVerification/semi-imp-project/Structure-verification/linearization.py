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

def Piola(d, lamda, mu):
    I = Identity(2)
    F = I + grad(d)
    E = 0.5*((F.T*F) - I)

    return F(lamda*tr(E)*I + 2*mu_s*E)

#Second Piola Kirchhoff Stress tensor
def Piola2(d_n, d_n0, lamda, mu):
    I = Identity(2)
    #F = I + grad(d_n)
    #E = 0.5*((F.T*F) - I)
    E = 1./2* ( 0.5*(grad(d_n).T + grad(d_n0).T + grad(d_n) + grad(d_n0)) \
    + 0.5*(grad(d_n).T*grad(d_n) + (grad(d_n0).T*grad(d_n0)) ) )

    return lamda*tr(E)*I + 2*mu_s*E

def Piola2_test(d_n, d_n0, lamda, mu):
    I = Identity(2)

    E = 1./2* ( 0.5*(grad(d_n).T + grad(d_n0).T + grad(d_n) + grad(d_n0)) \
    + 0.5*(grad(d_n) + grad(d_n0)).T * 0.5*(grad(d_n) + grad(d_n0)) )

    return lamda*tr(E)*I + 2*mu_s*E

def simple_lin(u, u_1, lamda, mu_s):
    I = Identity(2)
    E = 0.5*(grad(u) + grad(u).T + grad(u_1).T*grad(u))


    return lamda*tr(E)*I + 2*mu_s*E

def simple_lin_centered(d_n, d_n0, lamda, mu_s):
    I = Identity(2)
    E = 1./2* ( 0.5*(grad(d_n).T + grad(d_n0).T + grad(d_n) + grad(d_n0)) \
        + 0.5*(grad(d_n) + grad(d_n0)).T * 0.5*(grad(d_n) + grad(d_n0)) )
        #+ 0.5*(grad(d_n).T + grad(d_n_0).T) * 0.5*(grad(d_n) + grad(d_n0)) )
        #+ 0.5*(grad(d_n1).T*grad(d_n) + grad(d_n1).T*grad(d_n1) ) )


    return lamda*tr(E)*I + 2*mu_s*E

def piola2_adam_double(d_n, d_n1, d_n2, lamda, mu):
    I = Identity(2)
    E = 1./2* ( 0.5*(grad(d_n).T + grad(d_n1).T + grad(d_n) + grad(d_n1)) \
    + grad(3./2*d_n1 - 1./2*d_n2).T * 0.5*(grad(d_n) + grad(d_n1)) )
    #+ grad(3./2*d_n1 - 1./2*d_n2).T * grad(3./2*d_n1 - 1./2*d_n2) )

    return (lamda*tr(E)*I + 2*mu_s*E)

def piola2_adam_double2(d_n, d_n0, d_n1, w0, w_1, k, lamda, mu):
    I = Identity(2)
    E = 1./2* ( 0.5*(grad(d_n).T + grad(d_n0).T + grad(d_n) + grad(d_n0))  \
     + 0.5*(grad(d_n0 + k*(3./2*w0 - 1./2*w_1)).T + grad(d_n0).T) + 0.5*(grad(d_n) + grad(d_n0) ) )
    #+  grad(3./2*d_n1 -1./2*d_n2).T * 0.5*(grad(d_n) + grad(d_n1)) )

    return lamda*tr(E)*I + 2*mu_s*E

def test(u, u_1, lamda, mu_s):
    I = Identity(2)
    E = 0.5*( grad(u) + grad(u).T + grad(u_1).T*grad(u))
    E_1 = 0.5*( grad(u_1) + grad(u_1).T + grad(u_1).T*grad(u_1))
    F_1 = I + grad(u_1)
    F = I + grad(u)
    return (lamda*tr(0.5*(E + E_1))*I + 2*mu_s*0.5*(E + E_1))
    #return F_1*(lamda*tr(0.5*(E + E_1))*I + 2*mu_s*0.5*(E + E_1))


def solver(T, dt, space, implementation, betterstart):
    print "Solving for %s" % implementation

    t = 0
    k = Constant(dt)
    dis_x = []; dis_y = []; time = []
    plt.figure(1)

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
            G = rho_s/k*inner(w - w0, psi)*dx + inner(Piola2(d, d0, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx + dot(d - d0,phi)*dx - k*dot(0.5*(w + w0),phi)*dx

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

        if implementation == "test":
            wd = TrialFunction(VV)
            w, d = split(wd)

            G = rho_s/k*inner(w - w0, psi)*dx +inner(test(d, d0, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx + dot(d - d0 - k*0.5*(w - w0), phi)*dx

        if implementation =="A-B":
            wd = TrialFunction(VV)
            w, d = split(wd)
            G = rho_s/k*inner(w - w0, psi)*dx + inner(piola2_adam_double(d, d0, d_1, lamda, mu_s), grad(psi) )*dx \
            - inner(g, psi)*dx \
            + 1./k*inner(d - d0, phi)*dx - dot(0.5*(w + w0),phi)*dx# i#

        if implementation =="A-B2":
            wd = TrialFunction(VV)
            w, d = split(wd)
            G = rho_s/k*inner(w - w0, psi)*dx + inner(piola2_adam_double2(d, d0, d_1, w0, w_1, k, lamda, mu_s), grad(psi) )*dx \
            - inner(g, psi)*dx \
            + 1./k*inner(d - d0, phi)*dx - dot(0.5*(w + w0),phi)*dx# i#

        if implementation == "simple_lin":
            wd = TrialFunction(VV)
            w, d = split(wd)

            G = rho_s/k*inner(w - w0, psi)*dx +inner(simple_lin_centered(d, d0, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx \
            + 1./k*inner(d - d0, phi)*dx - dot(0.5*(w + w0),phi)*dx

        if implementation == "Piccard":
            # Define variational problem for Picard iteration
            wd = TrialFunction(VV)
            w, d = split(wd)
            wd_k = Function(VV)
            w_k, d_k = wd_k.split(VV)
            #G = rho_s/k*inner(w - w0, psi)*dx +inner(test2(d, d_k, d0, lamda, mu_s), grad(psi))*dx \
            #- inner(g, psi)*dx + dot(d - d0 - k*0.5*(w - w0), phi)*dx
            #org
            G = rho_s/k*inner(w - w0, psi)*dx +inner(simple_lin(d, d_k, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx + dot(d - d0 - k*w, phi)*dx

        if implementation =="C-N":
            #G = rho_s/k*inner(w - w0, psi)*dx + 0.5*inner(Piola2(d, d0, lamda, mu_s) + Piola2(d0, lamda, mu_s), grad(psi))*dx \
            G = rho_s/k*inner(w - w0, psi)*dx + inner(Piola2(d, d0, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx + dot(d - d0, phi)*dx - k*dot(0.5*(w + w0), phi)*dx

        if implementation =="C-N2":
            G = rho_s/k*inner(w - w0, psi)*dx + inner(Piola2_test(d, d0, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx + dot(d - d0, phi)*dx - k*dot(0.5*(w + w0), phi)*dx




    #dis_file = File("results/x_direction.pvd")
        if implementation == ("C-N" or "C-N2"):
            tic()
            print "HERE"
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
            print "Here piccard"
            tic()
            # Picard iterations
            a = lhs(G); L = rhs(G)
            #A = assemble(a);
            b = None
            wd_sol = Function(VV)
            maxiter = 10       # max no of iterations allowed
            tol = 1.0E-7        # tolerance

            while t <= T:
                eps = 1.0           # error measure ||u-u_k||
                iter = 0            # iteration counter
                #b = assemble(L, tensor=b)
                while eps > tol and iter < maxiter:
                    iter += 1
                    solve(a == L, wd_sol, bcs)
                    #A = assemble(a)
                    #[bc.apply(A, b) for bc in bcs]
                    #solve(A, wd_sol.vector(), b)
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


        else:
            print "HERE!!! , else imp %s " % implementation
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

                #print 'norm=%e' % ( eps)
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

    #if MPI.rank(mpi_comm_world()) == 0:
    #    if os.path.exists("./results/" + space + "/" + implementation + "/"+str(dt)) == False:
    #       os.makedirs("./results/" + space + "/" + implementation + "/"+str(dt))

    #    np.savetxt("./results/" + space + "/" + implementation + "/"+str(dt)+"/time.txt", time, delimiter=',')
    #    np.savetxt("./results/" + space + "/" + implementation + "/"+str(dt)+"/dis_y.txt", dis_y, delimiter=',')

    #    name = "./results/" + space + "/" + implementation + "/"+str(dt) + "/report.txt"  # Name of text file coerced with +.txt
    #    f = open(name, 'w')
    #    f.write("""Case parameters parameters\n """)
    #    f.write("""T = %(T)g\ndt = %(dt)g\nImplementation = %(implementation)s
    #    """ %vars())
    #    f.close()

    #plt.show()

comp_time = []
runs = []

T3 = {"space": "mixedspace", "implementation": "test", "T": 0.5, "dt": 0.005, "betterstart": False}; runs.append(T3)

PI3 = {"space": "mixedspace", "implementation": "Piccard", "T": 1.0, "dt": 0.002, "betterstart": False}; runs.append(PI3)

SI2 = {"space": "mixedspace", "implementation": "simple_lin", "T": 0.27, "dt": 0.001, "betterstart": False}; runs.append(SI2)

AB1 = {"space": "mixedspace", "implementation": "A-B", "T": 0.23, "dt": 0.001, "betterstart": False}; runs.append(AB1)
AB2 = {"space": "mixedspace", "implementation": "A-B2", "T": 0.015, "dt": 5E-5, "betterstart": False}; runs.append(AB2)
#CN2 = {"space": "mixedspace", "implementation": "C-N2", "T": 1.0, "dt": 0.005, "betterstart": False}; runs.append(CN2)
#CN2 = {"space": "mixedspace", "implementation": "C-N", "T": 0.015, "dt": 5E-5, "betterstart": False}; runs.append(CN2)


for r in runs:
    print r["implementation"], r["betterstart"]
    solver(r["T"], r["dt"], r["space"], r["implementation"], r["betterstart"])

for i in range(len(runs)):
    print "%s -- > CPU TIME %f" % (runs[i]["implementation"], comp_time[i])

    #plt.show()


#plt.figure(1)
#plt.title("implementation %s, x-dir" % (implementation))
#plt.plot(time,dis_x,);title; plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
#plt.savefig("run_x_imp%s.jpg" % (implementation))
#plt.show()
