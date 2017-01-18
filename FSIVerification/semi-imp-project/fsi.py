
from dolfin import *
from Utils.argpar import *
from Fluid_solver.projection import *
parameters['allow_extrapolation']=True

args = parse()
v_deg = args.v_deg
p_deg = args.p_deg
d_deg = args.d_deg

mesh = Mesh("fluid_new.xml")
plot(mesh, interactive=True)

for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break
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

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"
domains = CellFunction("size_t", mesh)
domains.set_all(1)
Bar_area.mark(domains, 2) #Overwrites structure domain
dx = Measure("dx", subdomain_data = domains)
#plot(domains,interactive = True)
dx_f = dx(1, subdomain_data = domains)
dx_s = dx(2, subdomain_data = domains)

dt = 0.1
mu = 1
rho = 1
k = Constant(dt)

#Fluid properties
rho_f   = Constant(1.0E3)
mu_f    = Constant(1.0)
nu = Constant(mu_f/rho_f)

#Structure properties
rho_s = 1.0E3
mu_s = 2.0E6
nu_s = 0.4
E_1 = 1.4E6
lamda_s = nu_s*2*mu_s/(1-2*nu_s)
g = Constant((0,-2*rho_s))


############## Define FunctionSpaces ##################

V = VectorFunctionSpace(mesh, "CG", 2)
V1 = VectorFunctionSpace(mesh, "CG", 1)
P = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace([V, V])

u = TrialFunction(V)
phi = TestFunction(V)
w = TrialFunction(V)

p = TrialFunction(P)
q = TestFunction(P)

vd = Function(W)
v, d = split(vd)

vd0 = Function(W)
# Make a deep copy to create two new Functions u and p (not subfunctions of W)
#Must be done in solve as well to redo step 0
v0, d0 = vd0.split(deepcopy=True) #

vd1 = Function(W)
v_1, d_1 = vd1.split(deepcopy=True)

############## Define BCS ##################
#BOUNDARY CONDITIONS

Um = 2.0
H = 0.41
L = 2.5
# "
inlet = Expression(("(1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2))*(1-cos(t*pi/2.0))/2.0" \
,"0"), t = 0.0, Um = Um, H = H)

#Fluid velocity conditions
u_inlet  = DirichletBC(V, inlet, boundaries, 3)
u_wall   = DirichletBC(V, ((0.0, 0.0)), boundaries, 2)
u_circ   = DirichletBC(V, ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
u_bar    = DirichletBC(V, ((0.0, 0.0)), boundaries, 5) #No slip on geometry in fluid

bcs_u = [u_inlet, u_wall, u_circ, u_bar]

#Pressure Conditions
p_out = DirichletBC(P, 0, boundaries, 4)

bcs_p = [p_out]

#Mesh velocity conditions
w_wall    = DirichletBC(V, ((0.0, 0.0)), boundaries, 2)
w_inlet   = DirichletBC(V, ((0.0, 0.0)), boundaries, 3)
w_outlet  = DirichletBC(V, ((0.0, 0.0)), boundaries, 4)
w_circle  = DirichletBC(V, ((0.0, 0.0)), boundaries, 6)
#w_barwall = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 7)
#w_bar     = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 5)

bcs_w = [w_wall, w_inlet, w_outlet, w_circle]

# Deformation conditions
d_inlet   = DirichletBC(W.sub(1), ((0, 0)), boundaries, 3)
d_wall    = DirichletBC(W.sub(1), ((0, 0)), boundaries, 2)
d_out     = DirichletBC(W.sub(1), ((0, 0)), boundaries, 4)
d_circ    = DirichletBC(W.sub(1), ((0, 0)), boundaries, 6)
d_barwall = DirichletBC(W.sub(1), ((0, 0)), boundaries, 7)

bcs_d = [d_inlet, d_wall, d_out, d_circ, d_barwall]

# Deformation conditions TILDE
d_inlet_t   = DirichletBC(V, ((0, 0)), boundaries, 3)
d_wall_t    = DirichletBC(V, ((0, 0)), boundaries, 2)
d_out_t     = DirichletBC(V, ((0, 0)), boundaries, 4)
d_circ_t    = DirichletBC(V, ((0, 0)), boundaries, 6)
d_barwall_t = DirichletBC(V, ((0, 0)), boundaries, 7)

bcs_d_tilde = [d_inlet_t, d_wall_t, d_out_t, d_circ_t, d_barwall_t]


############## Step 0: Extrapolation of the fluid-structure interface
d_tilde = Function(V) #Solution vector of F_expo
F_expo = inner(u - d0 - k*(3./2*d0 - 1./2*d_1 ), phi)*dx

############## Step 1: Definition of new domain
w_next = Function(V)   #Solution w_n+1 of F_smooth
d_move = Function(V1)  #d_move used in ALE.move method

F_smooth = inner(w - 1./k*(d_tilde -  d0), phi)*dx_s + inner(grad(u), grad(phi))*dx_f

############## Step 2: ALE-advection-diffusion step (explicit coupling)
u0 = Function(V)      # Same as u_tilde_n
u_tent = Function(V)  # Tentative velocity: solution of F_tent

#Works if dx_f is replaced by dx, fix should be ident_zeros, not work
F_tent = (rho_f/k)*inner(u - u0, phi)*dx_f + rho_f*inner(grad(u)*(u0 - w_next), phi)*dx_f + \
     2.*nu*inner(eps(u), eps(phi))*dx_f #+ inner(u('-') - w('-'), phi('-'))*dS(5)

############## Step 3: Projection Step (implicit coupling) ITERATIVE PART

# Pressure update
p_press = Function(P) #Solution of F_press_upt, pressure
F_press_upt = inner(grad(p), grad(q))*dx_f \
- (1./k)*div(u_tent)*q*dx_f \
#+ inner(dot(u, n) - 1./k*dot(d_tilde - d0, n), q)*dS(5)
#+ inner(u('-')*n('-') - 1./k*(d_tilde('-') - d0('-')*n('-')), phi('-'))*dS(5)
#First iterative d_tilde is gues for d_n+1
#Question must interior facets be specified for bcs in step 3????

# Velocity update
u_vel = Function(V) #Solution of F_vel_upt, velocity
F_vel_upt = inner(u, v)*dx_f - inner(u_tent, v)*dx_f + dot(k*grad(p_press), v)*dx_f

#while dt < T:

#Step 0:
solve(lhs(F_expo) == rhs(F_expo), d_tilde, bcs_d_tilde)

#Step 1:
solve(lhs(F_smooth) == rhs(F_smooth), w_next, bcs_d_tilde)
"""
w_1 = interpolate(w_next, V1)
w_1.vector()[:] *= float(k)
d_move.vector()[:] += w_1.vector()[:]
ALE.move(mesh, d_move)
mesh.bounding_box_tree().build(mesh)
"""
#Step 2:
#solve(lhs(F_tent) == rhs(F_tent), u_tent, bcs_u)
A = assemble(lhs(F_tent)); L = assemble(rhs(F_tent))
A.ident_zeros()
#[bc.apply(A, b) for bc in bcs_u]
#[bc.apply(A, b) for bc in bcs_p]
#solve(A , u_tent.vector(), b)


print "Step 2 DONE"

#Step 3:
solve(lhs(F_press_upt) == rhs(F_press_upt), )

#count += 1
#u0.assign(u1)
#p0.assign(p1)
#t += dt
