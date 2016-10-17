from dolfin import *

mesh = Mesh("fluid_new.xml")
#plot(mesh,interactive=True)

V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # Displacement
V3 = VectorFunctionSpace(mesh, "CG", 1) # mesh velocity
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VVVQ = MixedFunctionSpace([V1, V2, V3, Q])

# BOUNDARIES

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

Um = 0.2
H = 0.41
L = 2.5
inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2) * (1 - cos(t*pi/2))/2"\
,"0"), t = 0.0, Um = Um, H = H)

#Boundary conditions fluid
u_inlet  = DirichletBC(VVVQ.sub(0), inlet, boundaries, 3)
u_nos  = DirichletBC(VVVQ.sub(0), ((0.0,0.0)), boundaries, 2)
u_circle = DirichletBC(VVVQ.sub(0), ((0.0,0.0)), boundaries, 6)
# maybe set p = 0 on the outlet?

# Boundary conditions solid
d_bar  = DirichletBC(VVVQ.sub(1), ((0.0,0.0)), boundaries, 7)

#Boundary conditions mesh velocity
w_wall    = DirichletBC(VVVQ.sub(2), ((0, 0)), boundaries, 2)
w_inlet   = DirichletBC(VVVQ.sub(2), ((0, 0)), boundaries, 3)
w_outlet  = DirichletBC(VVVQ.sub(2), ((0, 0)), boundaries, 4)
w_circle  = DirichletBC(VVVQ.sub(2), ((0, 0)), boundaries, 6)
w_barwall = DirichletBC(VVVQ.sub(2), ((0, 0)), boundaries, 7)

bcs = [u_inlet, u_nos, u_circle ,\
        d_bar,\
        w_wall, w_inlet, w_outlet, w_circle, w_barwall]

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"

domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bar_area.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
plot(domains,interactive = True)
dx_f = dx(1,subdomain_data=domains)
dx_s = dx(2,subdomain_data=domains)

# TEST TRIAL FUNCTIONS
phi, psi, beta,gamma = TestFunctions(VVVQ)
#u, w, p = TrialFunctions(VVQ)
udwp = Function(VVVQ)
u,d, w, p  = split(udwp)
u0d0w0p0 = Function(VVVQ)
u0,d0, w0, p0  = split(udwp)


dt = 0.01
k = Constant(dt)

#Fluid properties
rho_f   = Constant(1.0)
mu_f    = Constant(1)

#Structure properties
rho_s = 1.0E3
mu_s = 0.5E6
nu_s = 0.4
E_1 = 1.4E6
lamda_s = nu_s*2*mu_s/(1-2*nu_s)

def s_s_n_l(U):
    I = Identity(2)
    F = I + grad(U)
    E = 0.5*((F.T*F)-I)
    return lamda_s*tr(E)*I + 2*mu_s*E

def sigma_fluid(p, u): #NEWTONIAN FLUID
    return -p*Identity(2) + 2*mu_f * sym(grad(u))

#Fluid variational form

F_fluid =  (rho_f/k)*inner(u -u0,phi)*dx_f + rho_f*inner(grad(u)*(u - w), phi)*dx_f \
+ inner(sigma_fluid(p, u), grad(phi))*dx_f \
- inner(div(u),gamma)*dx_f

#Solid variational form with displacement

F_solid =rho_s*((1./k)*inner(w-w0,psi))*dx_s + rho_s*inner(dot(grad(0.5*(w+w0)),0.5*(w+w0)),psi)*dx + inner(0.5*(s_s_n_l(d)+s_s_n_l(d0)),grad(psi))*dx_s \
     - dot(d-d0,beta)*dx_s + k*dot(0.5*(w+w0),beta)*dx_s

F_Laplace = k*inner(grad(w),grad(psi))*dx_f  - inner(grad(d0),grad(psi))*dx_f
F_interface = -k*inner(grad(w('-'))*n('-'),psi('-'))*dS(5)\
            + inner(grad(d0('-'))*n('-'),psi('-'))*dS(5)

#F_interface = inner(s_s_n_l(d)*n,beta)*dS(5)+inner(sigma_fluid(u,p)*n,beta)*dS(5)

F = F_fluid + F_solid  + F_Laplace + F_interface

T = 1.0
t = 0.0

#u_file = File("mvelocity/velocity.pvd")
#u_file << u0

while t <= T:
    print "Time t = %.3f" % t

    if t < 2:
        inlet.t = t;
    if t >= 2:
        inlet.t = 2;

    #J = derivative(F, uwdp)

    solve(F == 0, udwp, bcs, solver_parameters={"newton_solver": \
    {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})


    #dis_x.append(d(coord)[0])
    #dis_y.append(d(coord)[1])
    u_,d_, w_, p_ = udwp.split(True)

    ALE.move(mesh,d_)
    mesh.bounding_box_tree().build(mesh)
    plot(mesh,mode="displacement")
    u0d0w0p0.assign(udwp)

    #dis_x.append(d0(coord)[0])
    #dis_y.append(d0(coord)[1])

    t += dt
