from fenics import *
import mshr
import numpy as np
set_log_active(False)
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys

mesh = Mesh("von_karman_street_FSI_structure.xml")
mesh=refine(mesh)
mesh=refine(mesh)

for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break


# SOLID PARAMETERS
CSM = 3
Pr = 0.4
mu_s = [0.5, 2.0, 0.5][CSM-1]*1e6
rho_s = [1.0, 1.0, 1.0][CSM-1]*1e3
lamda_s = 2*mu_s*Pr/(1-2.*Pr)

g = Constant((0,-2*rho_s))
dt = float(sys.argv[1])
k = Constant(dt)
#beta = Constant(0.25)
print "mu: %.f , rho: %.f" %(mu_s, rho_s)

I = Identity(2)
def F_(U):
	return (Identity(2) + grad(U))

def J_(U):
	return det(F_(U))
def E(U):
	return 0.5*(F_(U).T*F_(U)-I)

def S(U,lamda_s,mu_s):
	return (2*mu_s*E(U) + lamda_s*tr(E(U))*I)

def P1(U,lamda_s,mu_s):
	return F_(U)*S(U,lamda_s,mu_s)



V = VectorFunctionSpace(mesh, "CG", 1)
W = V*V
n = FacetNormal(mesh)
print "Dofs: ",W.dim(), "Cells:", mesh.num_cells()

ud = Function(W)
u, d = split(ud)

phi, psi = TestFunctions(W)
ud0 = Function(W)
u0, d0 = split(ud0)

BarLeftSide =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
BarLeftSide.mark(boundaries,1)

#plot(boundaries,interactive=True)


delta = 1
F_structure = (rho_s/k)*inner(u-u0,phi)*dx
#F_structure += inner(P1(d,lamda_s,mu_s), grad(phi))*dx
#F_structure += delta*((1.0/k)*inner(d-d0,psi)*dx - inner(u,psi)*dx)
F_structure -= inner(g, phi)*dx #+ inner(f2, psi)*dx
F_structure += inner(0.5*(P1(d,lamda_s,mu_s)+P1(d0,lamda_s,mu_s)), grad(phi))*dx
F_structure += delta*((1.0/k)*inner(d-d0,psi)*dx - inner(0.5*(u+u0),psi)*dx)


u_file = XDMFFile(mpi_comm_world(), "Structure_MMS_results/velocity.xdmf")
d_file = XDMFFile(mpi_comm_world(), "Structure_MMS_results/d.xdmf")

for tmp_t in [u_file, d_file]:
    tmp_t.parameters["flush_output"] = True
    tmp_t.parameters["multi_file"] = 1
    tmp_t.parameters["rewrite_function_mesh"] = False


bc1 = DirichletBC(W.sub(0), ((0,0)),boundaries, 1)
bc2 = DirichletBC(W.sub(1), ((0,0)),boundaries, 1)

bcs = [bc1,bc2]
dis_x = []
dis_y = []
time_list = []
t = 0
T = 10
while t <= T:
    time_list.append(t)
    print "Time: %.2f" %t
    solve(F_structure == 0, ud, bcs,solver_parameters={"newton_solver": \
    {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
    u, d = ud.split(True)
    ud0.assign(ud)
    #u0.assign(u_); d0.assign(d_)

    dis_x.append(d(coord)[0])
    dis_y.append(d(coord)[1])
    print "x: ", d(coord)[0]
    print "y: ", d(coord)[1]
    #plot(d,mode = "displacement")
    t += dt
print "Dofs: ",W.dim(), "Cells:", mesh.num_cells()

title = plt.title("CSM 3 displacement of point A")
plt.figure(1)
#plt.title("Eulerian Mixed, schewed Crank-Nic")
plt.plot(time_list,dis_x,);title; plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
plt.axis([8, 10, -0.03, 0.005])

#plt.savefig("run_x.jpg")
plt.show()
plt.figure(2)
#plt.title("Eulerian Mixed, schewed Crank-Nic")
plt.plot(time_list,dis_y);title;plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
plt.axis([8, 10, -0.14, 0.02])
#plt.savefig("run_y.jpg")
plt.show()
