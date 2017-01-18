from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from time import time
import sys

from parameters import *
from parse import *
from solvers import *

args = parse()
v_deg = args.v_deg
p_deg = args.p_deg
d_deg = args.d_deg
T = args.T
dt = args.dt
beta = args.beta
step = args.step
fig = False

FSI= args.FSI_number
U_in = U_in(FSI)
mu_s = mu_s(FSI)
rho_s = rho_s(FSI)
lamda_s = lamda_s(FSI,mu_s)

time0 = time()

parameters["allow_extrapolation"] = True
mesh = Mesh("mesh/fluid_new.xml")
if args.refiner == None:
    print "None"
else:
    for i in range(args.refiner):
        mesh = refine(mesh)

for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break


V1 = VectorFunctionSpace(mesh, "CG", v_deg) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", d_deg) # displacement
Q  = FunctionSpace(mesh, "CG", p_deg)       # Fluid Pressure

VVQ = MixedFunctionSpace([V1, V2, Q])
print "Dofs: ",VVQ.dim(), "Cells:", mesh.num_cells()
# BOUNDARIES

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))
Bar = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.21)) or near(x[1], 0.19) or near(x[0], 0.6 ) )
Circle =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  ))
Barwall =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

Allboundaries = DomainBoundary()

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"


boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
Barwall.mark(boundaries, 7)
#Bar_area.mark(boundaries, 8)
#plot(boundaries,interactive=True)


ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh)


domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bar_area.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
#plot(domains,interactive = True)
dx_f = dx(1,subdomain_data=domains)
dx_s = dx(2,subdomain_data=domains)


# "
#inlet = Expression(("(1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2))*(1-cos(t*pi/2.0))/2.0" \
#,"0"), t = 0.0, Um = Um, H = H)
class inlet(Expression):
	def __init__(self):
		self.t = 0
	def eval(self,value,x):
		value[0] = 0.5*(1-np.cos(self.t*np.pi/2))*1.5*U_in*x[1]*(H-x[1])/((H/2.0)**2)
		value[1] = 0
	def value_shape(self):
		return (2,)
def Eij(U):
	return sym(grad(U))# - 0.5*dot(grad(U),grad(U))

def F_(U):
	return (I + grad(U))

def J_(U):
	return det(F_(U))

def E(U):
	return 0.5*(F_(U).T*F_(U)-I)

def S(U):
	return (2*mu_s*E(U) + lamda_s*tr(E(U))*I)

def P1(U):
	return F_(U)*S(U)

def sigma_f(v,p):
	return 2*mu_f*sym(grad(v)) - p*Identity(2)

def sigma_s(u):
	return 2*mu_s*sym(grad(u)) + lamda_s*tr(sym(grad(u)))*I

def sigma_f_hat(v,p,u):
	return J_(u)*sigma_f(v,p)*inv(F_(u)).T

inlet = inlet()
#Fluid velocity conditions
u_inlet  = DirichletBC(VVQ.sub(0), inlet, boundaries, 3)
u_wall   = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 2)
u_circ   = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
u_bar    = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 5) #No slip on geometry in fluid
u_barwall= DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

#displacement conditions:
d_wall    = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 2)
d_inlet   = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 3)
d_outlet  = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 4)
d_circle  = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 6)
d_barwall = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

#Pressure Conditions
p_out = DirichletBC(VVQ.sub(2), 0, boundaries, 4)

#Assemble boundary conditions
bcs = [u_inlet, u_wall, u_circ, u_barwall,\
       d_wall, d_inlet, d_outlet, d_circle,d_barwall,\
       p_out]#,p_bar]

# TEST TRIAL FUNCTIONS
phi, psi, gamma = TestFunctions(VVQ)

udp = Function(VVQ)
u, d, p  = split(udp)

udp0 = Function(VVQ)
udp_res = Function(VVQ)
d0 = Function(V2)
d1 = Function(V2)
u0 = Function(V1)
p0 = Function(Q)

k = Constant(dt)

I = Identity(2)

delta = 1.0E10
h =  mesh.hmin()

# Fluid variational form
F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx_f \
        + rho_f*inner(J_(d)*inv(F_(d))*grad(u)*(u - ((d-d0)/k)), phi)*dx_f \
        + inner(sigma_f_hat(u,p,d), grad(phi))*dx_f \
        - inner(div(J_(d)*inv(F_(d).T)*u), gamma)*dx_f

"""F_fluid = (rho_f/k)*inner(J_(0.5*(d+d1))*(u - u0), phi)*dx_f \
        + rho_f*inner(J_(0.5*(d+d1))*inv(F_(0.5*(d+d1)))*grad(0.5*(u+u0))*(0.5*(u+u0) - ((d-d1)/k)), phi)*dx_f \
        + inner(sigma_f_hat(0.5*(u+u0),0.5*(p+p0),0.5*(d+d1)), grad(phi))*dx_f \
        - inner(div(J_(0.5*(d+d1))*inv(F_(0.5*(d+d1)).T)*0.5*(u+u0)), gamma)*dx_f"""

if v_deg == 1:
    F_fluid += - beta*h*h*inner(J_(d)*inv(F_(d).T)*grad(p),grad(gamma))*dx_f
    print "v_deg",v_deg

# Structure var form
F_structure = (rho_s/k)*inner(u-u0,phi)*dx_s + inner(P1(d),grad(phi))*dx_s
#F_structure = (rho_s/(k))*inner((u-u0),phi)*dx_s + inner(0.5*(P1(d)+P1(d1)),grad(phi))*dx_s

# Setting w = u on the structure using (d-d0)/k = w
F_w = delta*((1.0/k)*inner(d-d0,psi)*dx_s - inner(u,psi)*dx_s)
#F_w = delta*((1.0/k)*inner(d-d1,psi)*dx_s - inner(0.5*(u+u0),psi)*dx_s)

# laplace
F_laplace =  (1./k)*inner(d-d0,psi)*dx_f +inner(grad(d), grad(psi))*dx_f #- inner(grad(d)*n, psi)*ds
#F_laplace =  (1./k)*inner(d-d1,psi)*dx_f +inner(grad(0.5*(d-d1)), grad(psi))*dx_f #- inner(grad(d)*n, psi)*ds

F = F_fluid + F_structure + F_w + F_laplace

t = 0.0
time_list = []

u_file = File("new_results_CN_structure_2/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/velocity.pvd")
d_file = File("new_results_CN_structure_2/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/d.pvd")
p_file = File("new_results_CN_structure_2/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/pressure.pvd")

dis_x = []
dis_y = []
Drag = []
Lift = []
counter = 0
t = dt

while t <= T:
    print "Time t = %.5f" % t
    time_list.append(t)
    if t < 2:
        inlet.t = t;
    if t >= 2:
        inlet.t = 2;

    #Reset counters
    atol = 1e-6;rtol = 1e-6; max_it = 100; lmbda = 1.0;

    udp = Newton_manual(F, udp, bcs, atol, rtol, max_it, lmbda,udp_res,VVQ)

    u,d,p = udp.split(True)

    #plot(u)
    if counter%step==0:
        u_file << u
        d_file << d
        p_file << p

        Dr = -assemble((sigma_f_hat(u,p,d)*n)[0]*ds(6))
        Li = -assemble((sigma_f_hat(u,p,d)*n)[1]*ds(6))
        Dr += -assemble((sigma_f_hat(u('-'),p('-'),d('-'))*n('-'))[0]*dS(5))
        Li += -assemble((sigma_f_hat(u('-'),p('-'),d('-'))*n('-'))[1]*dS(5))
        Drag.append(Dr)
        Lift.append(Li)

        dsx = d(coord)[0]
        dsy = d(coord)[1]
        dis_x.append(dsx)
        dis_y.append(dsy)

        if MPI.rank(mpi_comm_world()) == 0:
            print "t = %.4f " %(t)
            print 'Drag/Lift : %g %g' %(Dr,Li)
            print "dis_x/dis_y : %g %g "%(dsx,dsy)

    u0.assign(u)
    d1.assign(d0)
    d0.assign(d)
    p0.assign(p)
    t += dt
    counter +=1
print "script time: ", time()-time0
plt.plot(time_list,dis_x); plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
plt.savefig("new_results_CN_structure_2/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/dis_x.png")
#plt.show()
plt.plot(time_list,dis_y);plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
plt.savefig("new_results_CN_structure_2/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/dis_y.png")
#plt.show()
plt.plot(time_list,Drag);plt.ylabel("Drag");plt.xlabel("Time");plt.grid();
plt.savefig("new_results_CN_structure_2/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/drag.png")
#plt.show()
plt.plot(time_list,Lift);plt.ylabel("Lift");plt.xlabel("Time");plt.grid();
plt.savefig("new_results_CN_structure_2/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/lift.png")
#plt.show()
