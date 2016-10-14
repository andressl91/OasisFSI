from dolfin import *

mesh = Mesh("von_karman_street_FSI_fluid.xml")
#plot(mesh,interactive=True)

V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # Mesh movement
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VVQ = MixedFunctionSpace([V1, V2, Q])

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
#plot(boundaries,interactive=True)


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


#Mesh velocity conditions
w_wall    = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 2)
w_inlet   = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 3)
w_outlet  = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 4)
w_cirlce  = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 6)
w_barwall = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 7)

#deformation condition
#d_barwall = DirichletBC(VVQ.sub(2), ((0, 0)), boundaries, 7)

#Pressure Conditions
p_out = DirichletBC(VVQ.sub(2), 0, boundaries, 4)

#Assemble boundary conditions
bcs = [u_inlet, u_wall, u_circ,  \
       w_wall, w_inlet, w_outlet]
# AREAS

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"

domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bar_area.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
#plot(domains,interactive = True)


# TEST TRIAL FUNCTIONS
phi, psi, gamma = TestFunctions(VVQ)
#u, w, p = TrialFunctions(VVQ)
uwp = Function(VVQ)
u, w, p  = split(uwp)

#uwp0 = Function(VVQ)
#u0, w0, d0, p0  = split(uwp)

u0 = Function(V1); d0 =  Function(V2)

dt = 0.01
k = Constant(dt)
#EkPa = '62500'
#E = Constant(float(EkPa))

#Fluid properties
rho_f   = Constant(1.0)
mu_f    = Constant(1)

#Structure properties
rho_s = 1.0E3
mu_s = 0.5E6
nu_s = 0.4
E_1 = 1.4E6
lamda_s = nu_s*2*mu_s/(1-2*nu_s)

print "Re = %f" % (Um/(mu_f/rho_f))


def s_s_n_l(U):
    I = Identity(2)
    F = I + grad(U)
    E = 0.5*((F.T*F)-I)
    return lamda_s*tr(E)*I + 2*mu_s*E

def sigma_fluid(p, u): #NEWTONIAN FLUID
    return -p*Identity(2) + 2*mu_f * sym(grad(u))

delta = 0.1
d = d0 + k*u

# Fluid variational form
F_fluid =  (rho_f/k)*inner(u -u0,phi)*dx(1) + rho_f*inner(grad(u)*(u - w), phi)*dx(1) \
+ inner(sigma_fluid(p, u), grad(phi))*dx(1) \
- inner(div(u),gamma)*dx(1)

# Structure Variational form

F_structure = (rho_s/k)*inner(u-u0,phi)*dx(2) + rho_s*inner(dot(grad(0.5*(u+u0)),0.5*(u+u0)),phi)*dx(2) + inner(0.5*(s_s_n_l(d)+s_s_n_l(d0)),grad(phi))*dx(2) \



#Boundary condition
F_interface = inner(sigma_fluid(p,u)*n,phi)*dS(5) + inner(s_s_n_l(d)*n,phi)*dS(5)

#Laplace
F_laplace = k*inner(grad(w),grad(psi))*dx(1)-inner(grad(d0),grad(psi))*dx(1)

F_last = (1./delta)*inner(u,psi)*dx(2) - (1./delta)*inner(w,psi)*dx(2)



F = F_fluid + F_structure + F_interface + F_laplace + F_last

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

    #J = derivative(F, uwdp)

    solve(F == 0, uwp, bcs, solver_parameters={"newton_solver": \
    {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})


    #dis_x.append(d(coord)[0])
    #dis_y.append(d(coord)[1])

    u_, w_, p_ = uwp.split(True)
    u0.assign(u_)
    #w0.assign(w_) #Assigning new mesh velocity
    #d0.assign(d)
    w_.vector()[:] *= float(k)
    d0.vector()[:] += w_.vector()[:]
    #d0.assign(d)
    ALE.move(mesh,w_)
    mesh.bounding_box_tree().build(mesh)
    plot(mesh,mode="displacement")
    #dis_x.append(d0(coord)[0])
    #dis_y.append(d0(coord)[1])

    t += dt
