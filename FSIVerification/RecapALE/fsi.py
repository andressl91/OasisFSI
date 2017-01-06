from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import os
from argpar import parse
from postproses import postpro
from solvers import Newton_manual
parameters['allow_extrapolation']=True

args = parse()
v_deg = args.v_deg
p_deg = args.p_deg
d_deg = args.d_deg
theta = args.theta
discr = args.discr
T = args.T
dt = args.dt
fig = False


mesh = Mesh("fluid_new.xml")
m = "fluid_new.xml"

# For refining of mesh
if args.refiner != None:
    for i in range(args.refiner):
        mesh = refine(mesh)

for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break

# VectorFunctionSpaces
V1 = VectorFunctionSpace(mesh, "CG", v_deg) # Velocity
V2 = VectorFunctionSpace(mesh, "CG", d_deg) # Structure deformation
Q  = FunctionSpace(mesh, "CG", p_deg)       # Fluid Pressure
VVQ = MixedFunctionSpace([V1,V2,Q])

#Dofs and cells
U_dof = VVQ.dim()
mesh_cells = mesh.num_cells()

#Surfaces
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))
Bar = AutoSubDomain(lambda x: near(x[1], 0.21) or near(x[1], 0.19) or near(x[0], 0.6 ) )
Circle =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  ))
Barwall =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

#Areas
Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"

# FacetFunction for surfaces
boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
Barwall.mark(boundaries, 7)
#bou = File("bound.pvd")
#bou << boundaries

# Full inner geometry(flag + circle) for lift/drag integral
geometry = FacetFunction("size_t", mesh, 0)
Bar.mark(geometry, 1)
Circle.mark(geometry, 1)
Barwall.mark(geometry, 0)

#Are functions, for variational form
domains = CellFunction("size_t", mesh)
#domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(1)
Bar_area.mark(domains, 2) #Overwrites to structure domain

dx_f = dx(1,subdomain_data=domains)
dx_s = dx(2,subdomain_data=domains)
ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries) # For interface (interior)
n = FacetNormal(mesh)

# List for storing parameters
time = []; dis_x = []; dis_y = []
Lift = []; Drag = []

#Fluid properties
rho_f   = Constant(1000.0)
mu_f    = Constant(1.0)
nu_f = mu_f/rho_f

# Parameters
Um = 0.2
H = 0.41
L = 2.5
D = 0.1
k = Constant(dt)
t = dt

#Structure properties
rho_s = 1.0E3
mu_s = 0.5E6
nu_s = 0.4
E_1 = 1.4E6
lamda = nu_s*2*mu_s/(1 - 2*nu_s)

# velocity conditions
inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2) * (1 - cos(t*pi/2))/2"\
                        ,"0"), t = 0.0, Um = Um, H = H)

u_inlet   = DirichletBC(VVQ.sub(0), inlet,    boundaries, 3)
u_wall    = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 2)
u_circ    = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 6)
u_barwall = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 7)

# Deformation conditions
d_inlet   = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 3)
d_wall    = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 2)
d_out     = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 4)
d_circ    = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 6)
d_barwall = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 7)

# Pressure Conditions
p_out     = DirichletBC(VVQ.sub(2), 0, boundaries, 4)

# Assemble boundary conditions
bcs = [u_inlet, u_wall, u_circ, u_barwall, \
       d_inlet, d_wall, d_out, d_circ, d_barwall, \
       p_out]

# Functions
psi, gamma, eta = TestFunctions(VVQ)

udp = Function(VVQ)
u, d, p  = split(udp)

udp0 = Function(VVQ)
u0, d0, p0  = split(udp0)

I = Identity(2)

def F_(U):
	return (I + grad(U))

def J_(U):
	return det(F_(U))

def E(U):
	return 0.5*(F_(U).T*F_(U) - I)

def S(U): #St. Venant Kirchhoff material
	return (2*mu_s*E(U) + lamda*tr(E(U))*I)

def P1(U):
	return F_(U)*S(U)

def sigma_f(v, u, p):
	return 2*mu_f*sym(grad(v)) - p*Identity(2)

def sigma_f_map(v, u, p):
	return mu_f*(grad(v)*inv(F_(u)) + inv(F_(u)).T * grad(v).T) - p*Identity(2)

def sigma_s(u):
	return 2*mu_s*sym(grad(u)) + lamda*tr(sym(grad(u)))*I

def sigma_f_hat(v, u, p):
	return J_(u)*sigma_f_map(v, u, p)*inv(F_(u)).T

delta = 1.0E10
h =  mesh.hmin()

# Fluid variational form
F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), psi)*dx_f \
        + rho_f*inner(J_(d)*inv(F_(d))*grad(u)*(u - ((d - d0)/k)), psi)*dx_f \
        + inner(sigma_f_hat(u, d, p), grad(psi))*dx_f \
        - inner(inner(J_(d)*inv(F_(d)), grad(u).T), eta)*dx_f
        #- inner(div(J_(d)*inv(F_(d))*u), eta)*dx_f


# Structure var form
F_structure = (rho_s/k)*inner(u-u0,psi)*dx_s + inner(P1(d),grad(psi))*dx_s
#F_structure = (rho_s/(k*k))*inner(J_(0.5*(d+d1))*(d-2*d0+d1),phi)*dx_s + inner(0.5*(P1(d)+P1(d1)),grad(phi))*dx_s
#G =rho_s*((1./k)*inner(w-w0,psi))*dx  + rho_s*inner(dot(grad(0.5*(w+w0)),0.5*(w+w0)),psi)*dx \

# Setting w = u on the structure using (d-d0)/k = w
F_w = delta*((1.0/k)*inner(d - d0, gamma)*dx_s - inner(u, gamma)*dx_s)

# laplace
F_laplace = inner(grad(d), grad(gamma))*dx_f #+ (1./k)*inner(d - d0, gamma)*dx_f

F = F_fluid + F_structure + F_w + F_laplace

#Reset counters
d_up = TrialFunction(VVQ)
J = derivative(F, udp, d_up)
udp_res = Function(VVQ)

#Solver parameters
atol, rtol = 1e-6, 1e-6             # abs/rel tolerances
lmbda = 1.0                         # relaxation parameter
residual   = 1                      # residual (To initiate)
rel_res    = residual               # relative residual
max_it    = 15                      # max iterations
Iter = 0                            # Iteration counter

vel_file = File("./velocity/vel.pvd")
def_file = File("./deformation/def.pvd")

#[bc.apply(udp0.vector()) for bc in bcs]
u0, d0, p0  = udp0.split(True)
u0.rename("u", "velocity")
d0.rename("d", "deformation")
vel_file << u0
def_file << d0

Re = Um*D/nu_f
print "SOLVING FOR Re = %f" % Re #0.1 Cylinder diameter
tic()
count = 0

while t <= T:
    time.append(t)

    if t < 2:
        inlet.t = t;
    if t >= 2:
        inlet.t = 2;

    Newton_manual(F, udp, bcs, J, atol, rtol, max_it, lmbda, udp_res)

    u_, d_, p_  = udp.split(True)
    #if count % 10 == 0:
        #print "here"
    #u_.rename("u", "velocity")
    #vel_file << u_
    #d_.rename("d", "deformation")
    #def_file << d_

    u0, d0, p0  = udp0.split(True)

    #drag = -assemble((sigma_f_hat(u_, d_, p_)*n)[0]*ds(6)) - assemble((sigma_f_hat(u_('-'), d_('-'), p_('-'))* n('-'))[0]*dS(5))
    #lift = -assemble((sigma_f_hat(u_, d_, p_)*n)[1]*ds(6)) - assemble((sigma_f_hat(u_('-'), d_('-'), p_('-'))* n('-'))[1]*dS(5))

    drag = -assemble((sigma_f(u_, d_, p_)*n)[0]*ds(6)) - assemble((sigma_f(u_('-'), d_('-'), p_('-'))* n('-'))[0]*dS(5))
    lift = -assemble((sigma_f(u_, d_, p_)*n)[1]*ds(6)) - assemble((sigma_f(u_('-'), d_('-'), p_('-'))* n('-'))[1]*dS(5))

    #drag, lift =integrateFluidStress(p_, u_, geometry)
    #Drag.append(drag)
    #Lift.append(lift)
    #if MPI.rank(mpi_comm_world()) == 0:
    print "Time: ",t ," drag: ",drag, "lift: ",lift, "dis_x: ", d_(coord)[0], "dis_y: ", d_(coord)[1]

    udp0.assign(udp)

    dis_x.append(d_(coord)[0])
    dis_y.append(d_(coord)[1])
    #plot(d_, mode="displacement")

    count += 1
    t += dt

run_time = toc()

#if MPI.rank(mpi_comm_world()) == 0:
    #postpro(Lift, Drag, dis_x, dis_y, time, Re, m, U_dof, run_time, mesh_cells, case = 1)
