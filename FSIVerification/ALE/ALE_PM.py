from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Mesh
mesh = RectangleMesh(Point(0,0), Point(2, 1), 50, 50, "right")

# FunctionSpaces
V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # fluid displacement velocity
Q = FunctionSpace(mesh, "CG", 1)
VVQ = MixedFunctionSpace([V1, V2, Q])

# Boundaries
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.0)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 1.0) or near(x[1], 0)))

Allboundaries = DomainBoundary()
boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Inlet.mark(boundaries, 2)
Outlet.mark(boundaries, 3)
Wall.mark(boundaries, 4)
#plot(boundaries,interactive=True)

ds = Measure("ds", subdomain_data = boundaries)
n = FacetNormal(mesh)

# Boundary conditions
Wm = 0.01
inlet = Expression((("Wm","0")),Wm = Wm)

# Fluid velocity conditions
u_wall   = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 4)
u_inlet   = DirichletBC(VVQ.sub(0), inlet, boundaries, 2)

# Mesh velocity conditions
w_inlet   = DirichletBC(VVQ.sub(1), inlet, boundaries, 2)
w_wall    = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 4)

# Pressure Conditions
p_out = DirichletBC(VVQ.sub(2), 0, boundaries, 3)

#Assemble boundary conditions
bcs = [u_inlet,u_wall,\
       w_inlet,w_wall,\
       p_out]

# Test and Trial functions
phi, psi, gamma = TestFunctions(VVQ)
u, w, p = TrialFunctions(VVQ)
uwp = Function(VVQ)
#u, w, p = split(uwp)
u0 = Function(V1)
d0 = Function(V2)
dt = 0.01
k = Constant(dt)

# Fluid properties
rho_f   = Constant(1.0)
nu_f = Constant(1.0)
mu_f    = Constant(1.0)

def sigma_fluid(p, u): #NEWTONIAN FLUID
    I = Identity(2)
    F_ = I + grad(u)
    return -p*Identity(2) + mu_f *(grad(u)+grad(u).T)

def eps(u):
    return sym(grad(u))

#d = d0 + k*w
I = Identity(2)
F_ = I + grad(d0)
J_ = det(F_)

# Fluid variational form
F = J_*rho_f*((1.0/k)*inner(u - u0, phi) + inner(dot(inv(F_)*(u - w), grad(u0)), phi))*dx \
    + inner(J_*sigma_fluid(p, u)*inv(F_.T), grad(phi))*dx \
    - inner(div(J_*inv(F_.T)*u), gamma)*dx
    #- inner(J_*sigma_fluid(p,u)*inv(F_.T)*n, phi)*ds\

T = 1.0
t = 0.0
time = np.linspace(0,T,(T/dt)+1)

u_file = File("mvelocity/velocity.pvd")
w_file = File("mvelocity/w.pvd")
p_file = File("mvelocity/pressure.pvd")

while t <= T:
    solve(lhs(F)==rhs(F),uwp,bcs)
    u,w,p = uwp.split(True)
    plot(u)
    print "u-norm: ", norm(u)
    print "w-norm: ", norm(w)
    print "p-norm: ", norm(p)
    u0.assign(u)
    w.vector()[:] *= float(k)
    d0.vector()[:] += w.vector()[:]
    #d0.assign(d)
    u_file <<u
    t += dt
