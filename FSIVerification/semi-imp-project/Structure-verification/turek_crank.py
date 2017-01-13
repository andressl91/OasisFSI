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
def s_s_n_l(d):
    I = Identity(2)
    F = I + grad(d)
    E = 0.5*((F.T*F) - I)
    #J = det(F)
    return (lamda*tr(E)*I + 2*mu_s*E)

#Test for solving d directly
def piola_adam(d_n, d_n1, d_n2, lamda, mu):
    I = Identity(2)
    E = 1/4.*( (grad(d_n).T + grad(d_n2).T + grad(d_n) + grad(d_n2)) \
    + (3*grad(d_n1) - grad(d_n2)).T * (grad(d_n) + grad(d_n2)) )

    return (lamda*tr(E)*I + 2*mu_s*E)

#Test for solving d and
def piola_adam2(d_n, d_n1, d_n2, lamda, mu):
    I = Identity(2)
    E = 1/4.*( (grad(d_n).T + grad(d_n1).T + grad(d_n) + grad(d_n1)) \
    + (3*grad(d_n1) - grad(d_n2)).T * (grad(d_n) + grad(d_n1)) )

    return (lamda*tr(E)*I + 2*mu_s*E)


#Second order derivative
if implementation =="1":
    bcs = DirichletBC(V, ((0, 0)), boundaries, 1)
    psi = TestFunction(V)
    d = Function(V)
    d0 = Function(V)
    d1 = Function(V)

    #Testing proposed tensor
    G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx \
    + inner(piola_adam(d, d0, d1, lamda, mu_s), grad(psi))*dx \
    - inner(g,psi)*dx

    #original
    #G =rho_s*((1./k**2)*inner(d - 2*d0 + d1, psi))*dx - inner(g,psi)*dx\
    #+ inner(s_s_n_l(0.5*(d + d1)), grad(psi))*dx


#Split problem to two 1.order differential equations
if implementation =="2":
    bc1 = DirichletBC(VV.sub(0), ((0, 0)), boundaries, 1)
    bc2 = DirichletBC(VV.sub(1), ((0, 0)), boundaries, 1)
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

if implementation =="3":
    bc1 = DirichletBC(VV.sub(0), ((0,0)), boundaries, 1)
    bc2 = DirichletBC(VV.sub(1), ((0,0)), boundaries, 1)
    bcs = [bc1,bc2]
    psi, phi = TestFunctions(VV)
    wd = Function(VV)
    w, d = split(wd)
    w0 = Function(V); w_1 = Function(V)
    d0 = Function(V); d_1 = Function(V)

    G = rho_s/k*inner(w - w0, psi)*dx + inner(piola_adam2(d, d0, d_1, lamda, mu_s), grad(psi) )*dx \
    - inner(g,psi)*dx \
    + 1./k*inner(d - d0, phi)*dx - 0.5*inner(w + w0, phi)*dx


T = 4
t = 0


#dis_file = File("results/x_direction.pvd")

dis_x = []; dis_y = []; time = []
from time import sleep
while t <= T:
    if implementation == "1":
        solve(G == 0, d, bcs, solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
        d1.assign(d0)
        d0.assign(d)

        dis_x.append(d(coord)[0])
        dis_y.append(d(coord)[1])
        time.append(t)
        if MPI.rank(mpi_comm_world()) == 0:
            print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

    if implementation =="2":
        solve(G == 0, wd, bcs, solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
        w0d0.assign(wd)
        w0,d0 = w0d0.split(True)
        w,d = wd.split(True)

        dis_x.append(d(coord)[0])
        dis_y.append(d(coord)[1])
        time.append(t)
        if MPI.rank(mpi_comm_world()) == 0:
            print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

    if implementation =="3":
        solve(G == 0, wd, bcs, solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
        w, d = wd.split(True)
        d_1.assign(d0)
        d0.assign(d)
        w0.assign(w)

        dis_x.append(d0(coord)[0])
        dis_y.append(d0(coord)[1])
        time.append(t)
        if MPI.rank(mpi_comm_world()) == 0:
            print "Time: ",t #,"dis_x: ", d(coord)[0], "dis_y: ", d(coord)[1]

    t += dt


title = plt.title("Double space")

plt.figure(1)
plt.title("implementation %s, x-dir" % (implementation))
plt.plot(time,dis_x,);title; plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
plt.savefig("run_x_imp%s.jpg" % (implementation))
#plt.show()
plt.figure(2)
plt.title("implementation %s, y_dir" % (implementation))
plt.plot(time,dis_y);title;plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
plt.savefig("run_y_imp%s.jpg" % (implementation))
#plt.show()
