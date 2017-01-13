from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
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
dt = float(sys.argv[1])
k = Constant(dt)
beta = Constant(0.25)

#############
implementation = "3"
#############


#Second Piola Kirchhoff Stress tensor
def Piola2(d, lamda, mu):
    I = Identity(2)
    F = I + grad(d)
    E = 0.5*((F.T*F) - I)

    return (lamda*tr(E)*I + 2*mu_s*E)

def simple_lin(d_n, d_n1, lamda, mu):
    I = Identity(2)
    E = 1./2 * (grad(d_n) + grad(d_n).T + grad(d_n1)*grad(d_n).T)

    return lamda*tr(E)*I + 2*mu_s*E

def piola_adam_single(d_n, d_n1, d_n2, lamda, mu):
    I = Identity(2)
    E = 1./2* ( 0.5*(grad(d_n).T + grad(d_n2).T + grad(d_n) + grad(d_n2)) \
    +  1./2*(grad(d_n) + grad(d_n2)) * (3./2*grad(d_n1).T - 1./2*grad(d_n2).T) )

    return (lamda*tr(E)*I + 2*mu_s*E)

def piola2_adam_double(d_n, d_n1, d_n2, lamda, mu):
    I = Identity(2)
    E = 1./2* ( 0.5*(grad(d_n).T + grad(d_n1).T + grad(d_n) + grad(d_n1)) \
    +  1./2*(grad(d_n) + grad(d_n1)) * (3./2*grad(d_n1).T - 1./2*grad(d_n2).T) )

    return (lamda*tr(E)*I + 2*mu_s*E)

def solver(T, space, implementation, count, betterstart):

    t = 0
    dis_x = []; dis_y = []; time = []

    if space == "singlespace":
            bcs = DirichletBC(V, ((0, 0)), boundaries, 1)
            psi = TestFunction(V)
            d = Function(V)
            d0 = Function(V)
            d1 = Function(V)
            if betterstart == True:
                G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                + 0.5*inner(Piola2(d, lamda, mu_s) + Piola2(d1, lamda, mu_s), grad(psi))*dx

                while t < 2*dt:
                    solve(G == 0, d, bcs, solver_parameters={"newton_solver": \
                    {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
                    d1.assign(d0)
                    d0.assign(d)

                    dis_x.append(d(coord)[0])
                    dis_y.append(d(coord)[1])
                    time.append(t)
                    if MPI.rank(mpi_comm_world()) == 0:
                        print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

                    print "HERERERE"
                    t += dt

            if implementation == "C-N":
                print "CN"
                G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                + 0.5*inner(Piola2(d, lamda, mu_s) + Piola2(d1, lamda, mu_s), grad(psi))*dx

            if implementation == "simple_lin":
                G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                + inner(simple_lin(d, d0, lamda, mu_s), grad(psi))*dx

            if implementation == "A-B":
                G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
                + inner(piola_adam_single(d, d0, d1, lamda, mu_s), grad(psi))*dx


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
            - inner(g, psi)*dx + dot(d - d0,phi)*dx - k*dot(0.5*(w + w0),phi)*dx

            while t < 2*dt:
                print "HERE"
                solve(G == 0, wd, bcs, solver_parameters={"newton_solver": \
                {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})

                wd_1.assign(wd0)
                wd0.assign(wd)
                w_1, d_1 = wd_1.split(True)
                w0, d0 = wd0.split(True)
                w, d = wd.split(True)

                dis_x.append(d0(coord)[0])
                dis_y.append(d0(coord)[1])
                time.append(t)
                t += dt

                if MPI.rank(mpi_comm_world()) == 0:
                    print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

        if implementation =="C-N":
            G = rho_s/k*inner(w - w0, psi)*dx + 0.5*inner(Piola2(d, lamda, mu_s) + Piola2(d0, lamda, mu_s), grad(psi))*dx \
            - inner(g, psi)*dx + dot(d-d0,phi)*dx - k*dot(0.5*(w+w0),phi)*dx

        if implementation == "simple_lin":
            G = rho_s/k*inner(w - w0, psi)*dx + inner(simple_lin(d, d0, lamda, mu_s), grad(psi) )*dx \
            - inner(g, psi)*dx \
            + 1./k*inner(d - d0, phi)*dx - inner(w, phi)*dx

        if implementation =="A-B":
            G = rho_s/k*inner(w - w0, psi)*dx + inner(piola2_adam_double(d, d0, d_1, lamda, mu_s), grad(psi) )*dx \
            - inner(g, psi)*dx \
            + 1./k*inner(d - d0, phi)*dx - k*dot(0.5*(w + w0),phi)*dx

    #dis_file = File("results/x_direction.pvd")

    from time import sleep

    if space == "singlespace":
        tic()
        while t <= T:
            solve(G == 0, d, bcs, solver_parameters={"newton_solver": \
            {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
            d1.assign(d0)
            d0.assign(d)

            dis_x.append(d(coord)[0])
            dis_y.append(d(coord)[1])
            time.append(t)

            t += dt
            if MPI.rank(mpi_comm_world()) == 0:
                print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]
        comp_time.append(toc())
    if space == "mixedspace":
        while t <= T:
            tic()
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

    plt.figure(count)
    plt.title("implementation %s, y_dir" % (implementation))
    plt.plot(time,dis_y); plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
    plt.savefig("%s_%s_Ydef.jpg" % (space, implementation))
    #plt.show()

#space = ["singlespace"]
space = ["singlespace"]
implementation = ["simple_lin"]
#implementation = ["simple_lin"]
#implementation = ["A-B"]


comp_time = []
T = 1.0
count = 1
for s in space:
    for i in implementation:
        solver(T, s, i, count, betterstart = True)
        count += 1

for i in range(len(space)):
    for j in range(len(implementation)):
        print "CPU TIME %.3f" % comp_time[i*len(implementation) + j]

    #plt.show()

"""
plt.figure(1)
plt.title("implementation %s, x-dir" % (implementation))
plt.plot(time,dis_x,);title; plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
plt.savefig("run_x_imp%s.jpg" % (implementation))
#plt.show()
"""
