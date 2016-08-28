from dolfin import *
import numpy as np

#SJEKK Variert inflow, variatonal form, iteratons vs no it

mesh = Mesh("von_karman_street_FSI_fluid.xml")
#plot(mesh,interactive=True)

V = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

U_dof = V.dim()
mesh_cells = mesh.num_cells()

VVQ = MixedFunctionSpace([V,Q])

# BOUNDARIES

Inlet  = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 2.5))
Walls  = AutoSubDomain(lambda x: "on_boundary" and near(x[1],0) or near(x[1], 0.41))

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
DomainBoundary().mark(boundaries, 10)
Inlet.mark(boundaries, 2)
Outlet.mark(boundaries, 3)
Walls.mark(boundaries, 4)


ds = Measure("ds", subdomain_data = boundaries)
n = FacetNormal(mesh)
plot(boundaries,interactive=True)

#BOUNDARY CONDITIONS

Um = 0.2
H = 0.41

##UNSTEADY FLOW
#inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / (H/2.0*H/2.0) * (1 - cos(pi/2*t))/2"\
#,"0"), t = 0.0, Um = Um, H = H)
#STEADY FLOW
inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / (H/2.0*H/2.0)"\
,"0"), Um = Um, H = H)

u_inlet = DirichletBC(VVQ.sub(0), inlet, boundaries, 2)
nos_geo = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 1)
nos_wall = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 4)

p_out = DirichletBC(VVQ.sub(1), 0, boundaries, 3)

bcs = [u_inlet, nos_geo, nos_wall, p_out]


# TEST TRIAL FUNCTIONS
phi, eta = TestFunctions(VVQ)
u ,p = TrialFunctions(VVQ)

u0 = Function(V1)
u1 = Function(V1)

dt = 0.01
k = Constant(dt)

#Physical parameter

rho_f = 1000.0
nu = 0.001
mu_f = rho_f*nu

def sigma_fluid(p,u):
    return -p*Identity(2) + mu_f * (grad(u) + grad(u).T)#sym(grad(u))
	#Change sym_grad(u) to grad(u)

# Fluid variational form
F = (1./k)*inner(u - u1, phi)*dx \
    + inner(dot(u0, grad(u)), phi) * dx \
    + (1./rho_f)*inner(sigma_fluid(p,u), grad(phi))*dx - inner(div(u),eta)*dx

a = lhs(F)
L = rhs(F)


def integrateFluidStress(u, p):

    eps   = 0.5*(grad(u) + grad(u).T)
    sig   = -p*Identity(2) + 2.0*mu_f*eps

    traction  = dot(sig, n)

    forceX  = traction[0]*ds(1)
    forceY  = traction[1]*ds(1)
    fX      = assemble(forceX)
    fY      = assemble(forceY)

    return fX, fY

T = 10.0
t = 0.0
up = Function(VVQ)

Drag = []
Lift = []
time = []

#A = assemble(a)
Re = Um*(0.05*2)/(mu_f/rho_f)
print "SOLVING FOR Re = %f" % Re #0.1 Cylinder diameter
print "DOF = %f,  cells = %f" % (U_dof, mesh_cells)
while t < T:

	time.append(t)

	b = assemble(L)
	eps = 10
	k_iter = 0
	max_iter = 5
	while eps > 1E-6 and k_iter < max_iter:
		A = assemble(a)
		A.ident_zeros()
		[bc.apply(A,b) for bc in bcs]
		solve(A, up.vector(), b)
		u_, p_ = up.split(True)
		eps = errornorm(u_,u0,degree_rise=3)
		k_iter += 1
		print "k: ",k_iter, "error: %.3e" %eps
		u0.assign(u_)

	#solve(a == L, up, bcs)
	#u_,p_ = up.split(True)

	drag, lift =integrateFluidStress(u_, p_)
	if MPI.rank(mpi_comm_world()) == 0:
		print "Time: ",t ," drag: ",drag, "lift: ",lift
	Drag.append(drag)
	Lift.append(lift)

	u1.assign(u_)
	t += dt

if MPI.rank(mpi_comm_world()) == 0:
	import matplotlib.pyplot as plt
	plt.figure(1)
	plt.title("LIFT \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
	plt.xlabel("Time Seconds")
	plt.ylabel("Lift force Newton")
	plt.plot(time, Lift)
	plt.figure(2)
	plt.title("Drag \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
	plt.xlabel("Time Seconds")
	plt.ylabel("Lift force Newton")
	plt.plot(time, Drag)

	plt.show()

#Drag[0] = 0; Lift[0] = 0
#print "max Drag",max(Drag), "max Lift",max(Lift)
