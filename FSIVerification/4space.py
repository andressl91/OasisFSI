from dolfin import *

mesh = Mesh("von_karman_street_FSI_fluid.xml")
#plot(mesh,interactive=True)

V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # Mesh movement
V3 = VectorFunctionSpace(mesh, "CG", 1) # Displacement
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VVQ = MixedFunctionSpace([V1, V2, V3, Q])

# BOUNDARIES

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))
Bar = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.21)) or near(x[1], 0.19) or near(x[0], 0.6 ) )
Circle =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  ))
Barwall =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

Allboundaries = DomainBoundary()

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
Barwall.mark(boundaries, 7)
plot(boundaries,interactive=True)


ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh)

#BOUNDARY CONDITIONS

Um = 0.2
H = 0.41
L = 2.5
inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2) * (1 - cos(t*pi/2))/2"\
,"0"), t = 0.0, Um = Um, H = H)

#Fluid velocity conditions
u_inlet  = DirichletBC(VVQ.sub(0), inlet, boundaries, 3)
u_wall   = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 2)
u_circ   = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 6) #No slip on geometry in fluid
u_bar    = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 5)

#Mesh velocity conditions
w_wall    = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 2)
w_inlet   = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 3)
w_outlet  = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 4)
w_cirlce  = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 6)
w_barwall = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 7)

#deformation condition
d_barwall = DirichletBC(VVQ.sub(2), ((0, 0)), boundaries, 7)

#Pressure Conditions
p_out = DirichletBC(VVQ.sub(3), 0, boundaries, 4)

#Assemble boundary conditions
bcs = [u_inlet, u_wall, u_circ, u_bar, \
       w_wall, w_inlet, w_outlet, \
       d_barwall]
# AREAS

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"

domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bar_area.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
plot(domains,interactive = True)


# TEST TRIAL FUNCTIONS
phi, psi, gamma, eta = TestFunctions(VVQ)
#u, w, p = TrialFunctions(VVQ)
uwp = Function(VVQ)
u, w, d, p  = split(uwp)

#uwp0 = Function(VVQ)
#u0, w0, d0, p0  = split(uwp)

u0 = Function(V1); w0 = Function(V2); d0 =  Function(V3)

dt = 0.01
k = Constant(dt)
#EkPa = '62500'
#E = Constant(float(EkPa))

#Fluid properties
rho_f   = Constant(100.0)
mu_f    = Constant(1)

#Structure properties
mu_s    = Constant(10E12)
rho_s   = Constant(10E6)
nu_s    = Constant(mu_s/rho_s)
lamda_s = nu_s*2.*mu_s/(1 - nu_s*2)

print "Re = %f" % (Um/(mu_f/rho_f))

def sigma_s(U1, U0): #NONLINEAR
    g = Constant(9.81)
    B = Constant((0, g*rho_s))
    T = Constant((0, 0))

    #F = Identity(2) + grad(0.5*(U1 - U0))
    F = Identity(2) + grad(U0)
    C = F.T*F

    E = 0.5 * (C - Identity(2))
    E = variable(E)
    W = lamda_s/2.*(tr(E))*(tr(E)) + mu_s * (tr(E*E))
    S = diff(W, E)
    P = F*S
    return P

def sigma_structure(d, d0): #HOOKES LAW
    return 2*mu_s*sym(grad(d)) + lamda_s*tr(sym(grad(d)))*Identity(2)

def s_s_n_l(U, U0):
    I = Identity(2)
    F = I + grad(U)
    E = 0.5*((F.T*F)-I)
    return lamda_s*tr(E)*I + 2*mu_s*E

def sigma_fluid(p, u): #NEWTONIAN FLUID
    return -p*Identity(2) + 2*mu_f * sym(grad(u))

# Fluid variational form
#F_fluid = ( (rho_f/k)*inner(u -u0,phi) + rho_f*inner(grad(u)*(u - w), phi) \
#+ inner(sigma_fluid(p, u), grad(phi)) \
#- inner(div(u),eta))*dx(1)

# Structure Variational form
#F_solid = ((rho_s/k)*inner(w - w0, psi))*dx(2) + rho_s*inner(dot(grad(w), w), psi)*dx(2)\
#+ inner(s_s_n_l(U),grad(psi))*dx(2) \

## Fernandez ######################
F_fluid = ( (rho_f/k)*inner(u -u0,phi) + rho_f*inner(grad(u)*(u - w), phi) \
- rho_f*inner(div(w)*u, phi) \
+ inner(sigma_fluid(p, u), grad(phi)) \
+ inner(div(u),eta))*dx(1)

F_solid = ((2.*rho_s/(k*k))*inner(d - d0 - k*w0, psi))*dx(2) + inner(sigma_s(d, d0),grad(psi))*dx(2)
###################################


## FENICS COURSE SOLID FORMULATION
#F_solid = rho_s*dot((w - w0), psi)*dx + dt*inner(sigma_structure(d, d0), grad(psi))*dx + dot((d - d0),gamma)*dx \
#    - dt*dot((w + w0)/2., gamma)*dx

# Mesh velocity function in fluid domain
F_meshvel = inner(grad(d),grad(gamma))*dx(1) - inner(grad(d("-"))*n("-"),gamma("-"))*dS(5)

F = F_fluid + F_solid #- F_meshvel

T = 0.1
t = 0.0

#u_file = File("mvelocity/velocity.pvd")
#u_file << u0

while t <= T:
    if MPI.rank(mpi_comm_world()) == 0:
        print "Time t = %.3f" % t

    if t < 2:
        inlet.t = t;
    if t >= 2:
        inlet.t = 2;

    J = derivative(F, uwp)

    problem = NonlinearVariationalProblem(F, uwp, bcs, J)
    solver  = NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm['newton_solver']['absolute_tolerance'] = 1E-8
    prm['newton_solver']['relative_tolerance'] = 1E-8
    prm['newton_solver']['maximum_iterations'] = 5
    prm['newton_solver']['relaxation_parameter'] = 1.0


    solver.solve()

    u_, w_, d_, p_ = uwp.split(True)
    u0.assign(u_)
    w0.assign(w_) #Assigning new mesh velocity
    d0.assign(d_)
    ALE.move(mesh, d_)
    mesh.bounding_box_tree().build(mesh)
    #uwp0.assign(uwp)

    t += dt
