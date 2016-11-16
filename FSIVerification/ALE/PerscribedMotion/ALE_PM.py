from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
set_log_active(True)
# Mesh
mesh = RectangleMesh(Point(0,0), Point(2, 1), 20, 20, "crossed")

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
plot(boundaries,interactive=True)

ds = Measure("ds", subdomain_data = boundaries)
n = FacetNormal(mesh)

# Boundary conditions
Wm = 0.1
#inlet = Expression((("Wm","0")),Wm = Wm,t=0)
inlet = Expression((("sin(pi*t)/3.0","0")),Wm = Wm,t=0)


# Fluid velocity conditions
class U_bc(Expression):
    def init(self,w):
        self.w = w
    def eval(self,value,x):
        #x_value, y_value = self.w.vector()[[x[0], x[1]]]
        value[0], value[1] = self.w(x)
        #value[0] = x_value
        #value[1] = y_value
    def value_shape(self):
        return (2,)

u_bc = U_bc(degree=2)

# Fluid velocity conditions
u_wall   = DirichletBC(VVQ.sub(0), u_bc, boundaries, 4)
u_inlet   = DirichletBC(VVQ.sub(0), inlet, boundaries, 2)


# Mesh velocity conditions
w_inlet   = DirichletBC(V2, inlet, boundaries, 2)
w_outlet  = DirichletBC(V2, ((0.0, 0.0)), boundaries, 3)

# Pressure Conditions
p_out = DirichletBC(VVQ.sub(1), 0, boundaries, 3)

#Assemble boundary conditions
bcs_w = [w_inlet, w_outlet]
bcs_u = [u_inlet, u_wall, p_out]

# Test and Trial functions
phi, gamma = TestFunctions(VVQ)
#u,p = TrialFunctions(VVQ)
psi = TestFunction(V2)
w = TrialFunction(V2)
up_ = Function(VVQ)
u, p = split(up_)
u0 = Function(V1)
d = Function(V2)
w_ = Function(V2)

dt = 0.02
k = Constant(dt)

# Fluid properties
rho_f   = Constant(1.0)
nu_f = Constant(1.0)
mu_f    = Constant(1.0)

def sigma_fluid(p, u): #NEWTONIAN FLUID
    #I = Identity(2)
    #F_ = I + grad(u)
    return -p*Identity(2) + mu_f *(inv(F_)*grad(u)+grad(u).T*inv(F_.T))

def eps(u):
    return sym(grad(u))

d = k*w_
I = Identity(2)
F_ = I + grad(d)
J_ = det(F_)

# Fluid variational form
F1 = J_*rho_f*((1.0/k)*inner(u - u0, phi) + inner(dot(inv(F_)*(u - w_), grad(u)), phi))*dx \
    + inner(J_*sigma_fluid(p, u)*inv(F_.T), grad(phi))*dx \
    - inner(div(J_*inv(F_.T)*u), gamma)*dx\
    - inner(J_*sigma_fluid(p,u)*inv(F_.T)*n, phi)*ds

# laplace d = 0
F2 =  k*(inner(grad(w), grad(psi))*dx - inner(grad(w)*n, psi)*ds)

T = 2.0
t = 0.0

u_file = File("mvelocity/velocity.pvd")
w_file = File("mvelocity/w.pvd")
p_file = File("mvelocity/pressure.pvd")
d_file = File("mvelocity/displacement.pvd")

#w_.vector()[:] *= float(k) # gives displacement to be used in ALE.move(w_)

#plot(w_,interactive=True)
time_array = np.linspace(0,T,(T/dt))
flux = []
while t <= T:
    print "Time: ",t
    inlet.t = t
    solve(lhs(F2)==rhs(F2), w_, bcs_w)
    u_bc.init(w_)

    solve(F1==0, up_, bcs_u,solver_parameters={"newton_solver": \
    {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
    u,p = up_.split(True)
    #print "u-w : ", assemble(dot(u-w_,n)*ds(4))
    plot(u)
    flux.append(assemble(J_*dot(u,n)*ds(3)))
    u0.assign(u)
    u_file << u
    #p_file << p
    #w_file << w_

    # To see the mesh move with a give initial w
    #ALE.move(mesh,w_)
    #plot(mesh)#,interactive = True)

    t += dt
print len(flux),len(time_array)
plt.plot(time_array,flux);plt.title("Flux, with N = 20"); plt.ylabel("Flux out");plt.xlabel("Time");plt.grid();
plt.show()
