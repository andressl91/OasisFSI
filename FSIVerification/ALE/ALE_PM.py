from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Mesh
mesh = RectangleMesh(Point(0,0), Point(2, 1), 50, 50, "right")

# FunctionSpaces
V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # fluid displacement velocity
Q = FunctionSpace(mesh, "CG", 1)
VVQ = MixedFunctionSpace([V1, Q])

# Boundaries
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0], 2.0)))
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
class U_bc(Expression):
    def init(self,w):
        self.w = w
    def eval(self,value,x):
        value[0] = w.vector()[x[0]]
        value[1] = w.vector()[x[1]]
    def value_shape(self):
        return (2,)

u_bc = U_bc()

# Mesh velocity conditions
w_inlet   = DirichletBC(V2, inlet, boundaries, 2)
w_outlet  = DirichletBC(V2, ((0.0, 0.0)), boundaries, 3)

# Pressure Conditions
p_out = DirichletBC(VVQ.sub(1), 0, boundaries, 3)

#Assemble boundary conditions
bcs_w = [w_inlet, w_outlet]

# Test and Trial functions
phi, gamma = TestFunctions(VVQ)
u,p = TrialFunctions(VVQ)
psi = TestFunction(V2)
w = TrialFunction(V2)
up_ = Function(VVQ)
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
    #I = Identity(2)
    #F_ = I + grad(u)
    return -p*Identity(2) + mu_f *(inv(F_)*grad(u)+grad(u).T*inv(F_.T))
#TODO: fluid spenningstensor med F og J

def eps(u):
    return sym(grad(u))

#d = d0 + k*w
I = Identity(2)
F_ = I + grad(d0)
J_ = det(F_)

# Fluid variational form
F1 = J_*rho_f*((1.0/k)*inner(u - u0, phi) + inner(dot(inv(F_)*(u - w), grad(u0)), phi))*dx \
    + inner(J_*sigma_fluid(p, u)*inv(F_.T), grad(phi))*dx \
    - inner(div(J_*inv(F_.T)*u), gamma)*dx\
    - inner(J_*sigma_fluid(p,u)*inv(F_.T)*n, phi)*ds\

# laplace d = 0
F2 =  k*(inner(grad(w), grad(psi))*dx - inner(grad(w)*n, psi)*ds)

T = 1.0
t = 0.0
time = np.linspace(0,T,(T/dt)+1)

u_file = File("mvelocity/velocity.pvd")
w_file = File("mvelocity/w.pvd")
p_file = File("mvelocity/pressure.pvd")

w_ = Function(V2)
solve(lhs(F2)==rhs(F2), w_, bcs_w)
u_bc.init(w_)
plot(w_,interactive=True)

# Fluid velocity conditions
u_wall   = DirichletBC(VVQ.sub(0), u_bc, boundaries, 4)
u_inlet   = DirichletBC(VVQ.sub(0), inlet, boundaries, 2)
bcs_u = [u_inlet,u_wall, p_out]

while t <= T:
    solve(lhs(F1)==rhs(F1), up_, bcs_u)
    u,p = up_.split(True)
    plot(w)
    print "u-norm: ", norm(u)
    print "w-norm: ", norm(w)
    print "p-norm: ", norm(p)
    print "flux out ", assemble(dot(u,n)*ds(3))
    u0.assign(u)
    w.vector()[:] *= float(k)
    d0.vector()[:] += w.vector()[:]

    #d0.assign(d)
    u_file <<u
    w_file <<w
    p_file <<p
    t += dt
