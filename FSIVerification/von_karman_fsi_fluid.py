from dolfin import *

mesh = Mesh("von_karman_street_FSI_fluid.xml")
#plot(mesh,interactive=True)

V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # Mesh movement
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VVQ = MixedFunctionSpace([V1,V2,Q])

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
inlet = Expression(("1.5*Um*4.0/0.1681*x[1]*(H-x[1])", "0"), t = 0.0, Um = Um, H = H)

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

#Pressure Conditions
p_out = DirichletBC(VVQ.sub(2), 0, boundaries, 4)

#Assemble boundary conditions
bcs = [u_inlet, u_wall, u_circ, u_bar, \
       w_wall, w_inlet, w_outlet, \
       p_out]
# AREAS

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"

domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bar_area.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
plot(domains,interactive = True)


# TEST TRIAL FUNCTIONS
phi, psi, eta = TestFunctions(VVQ)
u, w, p = TrialFunctions(VVQ)

u0 = Function(V1)
u1 = Function(V1)
w0 = Function(V2)
U1 = Function(V2)

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
lamda_s = nu_s*2*mu_s/(1 - nu_s*2)

print "Re = %f" % (Um/(mu_f/rho_f))

def sigma_s(t): #NONLINEAR
    g = Constant(9.81)
    B = Constant((0, g*rho_s))
    T = Constant((0, 0))

    F = Identity(2) + grad(0.5*(t))
    #F = Identity(2) + grad(0.5*(t))
    C = F.T*F

    E = 0.5 * (C - Identity(2))
    E = variable(E)
    W = lamda_s/2.*(tr(E))*(tr(E)) + mu_s * (tr(E*E))
    S = diff(W, E)
    P = F*S
    return P

def sigma_structure(d): #HOOKES LAW
    return 2*mu_s*sym(grad(d)) + lamda_s*tr(sym(grad(d)))*Identity(2)

def sigma_fluid(p,u): #NEWTONIAN FLUID
    return -p*Identity(2) + 2*mu_f * sym(grad(u))

# Fluid variational form
F = rho_f*((1./k)*inner(u-u1,phi)*dx(1) \
+ inner(grad(u0)*( u - w), phi) * dx(1)
+ inner(sigma_fluid(p,u),grad(phi))*dx(1)) - inner(div(u),eta)*dx(1)

#deformation formulation as a function of mesh velocity
U = U1 + w*k

# Structure Variational form
G = rho_s*(1./k*inner(w - w0, psi))*dx(2) +  inner(sigma_structure(U), grad(psi))*dx(2) \
#+ inner(u,phi)*dx(2) - inner(w,phi)*dx(2)

# Mesh movement, solving the equation laplace -nabla(grad(d)) = 0
H = inner(grad(U),grad(psi))*dx(1) - inner(grad(U("-"))*n("-"),psi("-"))*dS(5)

a = lhs(F) - lhs(G) - lhs(H)
L = rhs(F) - rhs(G) - rhs(H)

#F1 = F-G-H

T = 10.0
t = 0.0
uwp = Function(VVQ)

u_file = File("mvelocity/velocity.pvd")
u_file << u0
Lift = []; t_list = []
while t < T:
    #if MPI.rank(mpi_comm_world()) == 0:
    t_list.append(t)
    b = assemble(L)
    eps = 10
    k_iter = 0
    max_iter = 1
    while eps > 1E-6 and k_iter < max_iter:
        A = assemble(a)
        A.ident_zeros()
        [bc.apply(A,b) for bc in bcs]
        solve(A,uwp.vector(),b)
        u_,w_,p_ = uwp.split(True)
        eps = errornorm(u_,u0,degree_rise=3)
        k_iter += 1
        print "k: ",k_iter, "error: %.3e" %eps
        u0.assign(u_)
    w0.assign(w_) #Assigning new mesh velocity
    u1.assign(u_) #Assigning new fluid velocity
    w_.vector()[:] *= float(k) #Computing new deformation
    U1.vector = w_.vector()[:] #Applying new deformation, and updating
    ALE.move(mesh,w_)
    mesh.bounding_box_tree().build(mesh)


    #Calculate lift on bar and circle
    Force_lift = assemble(dot(sigma_fluid(p_, u_), n)[1]*ds(5) + \
                          dot(sigma_fluid(p_, u_), n)[1]*ds(6))
    Lift.append(Force_lift)

    u_file << u_
    print "Time:",t

    t += dt

plot(u_, interactive=True)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(t_list, Lift)
plt.show()

#plot(u_,interactive=True)
