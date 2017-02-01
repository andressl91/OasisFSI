from solvers import Newton_manual
from dolfin import *
from Utils.argpar import *
#from Fluid_solver.projection import *
parameters['allow_extrapolation']=True

args = parse()
v_deg = args.v_deg
p_deg = args.p_deg
d_deg = args.d_deg
dt = args.dt

mesh = Mesh("fluid_new.xml")
#plot(mesh, interactive=True)

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
#splot(boundaries,interactive=True)

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


k = Constant(dt)

#Fluid properties
rho_f   = Constant(1.0E3)
mu_f    = Constant(1.0)
nu = Constant(mu_f/rho_f)

#Structure properties
rho_s = Constant(1.0E3)
mu_s = Constant(0.5E6)
nu_s = Constant(0.4)
E_1 = 1.4E6
lamda_s = nu_s*2*mu_s/(1 - 2*nu_s)

############## Define FunctionSpaces ##################

U = VectorFunctionSpace(mesh, "CG", 2)
V1 = VectorFunctionSpace(mesh, "CG", 1)
P = FunctionSpace(mesh, "CG", 1)
V = MixedFunctionSpace([U, P])
W = MixedFunctionSpace([U, U])

u_ = TrialFunction(U) #Used for unknown in all linear steps 0, 1, 2
phi = TestFunction(U)
w = TrialFunction(U)

psieta = TestFunction(V)
psi, eta = split(psieta)

up = TrialFunction(V)
u, p = split(up)

up0 = Function(V)
u0, p0 = split(up0)
#u0, p0 = up0.split(deepcopy = True)

up_sol = Function(V)

ab = TestFunction(W)
alfa, beta = split(ab)
vd = Function(W)
v, d = split(vd)

vd0 = Function(W)
# Make a deep copy to create two new Functions u and p (not subfunctions of W)
#Must be done in solve as well to redo step 0
#v0, d0 = split(vd0) #

v0, d0 = vd0.split(deepcopy=True) #

vd1 = Function(W)
#v_1, d_1 = split(vd1)
v_1, d_1 = vd1.split(deepcopy=True)

############## Define BCS ##################
#BOUNDARY CONDITIONS

Um = 0.2
D = 0.1
H = 0.41
L = 2.5
# "
inlet = Expression(("(1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2))*(1-cos(t*pi/2.0))/2.0" \
,"0"), t = 0.0, Um = Um, H = H)

#Fluid velocity conditions
u_inlet  = DirichletBC(V.sub(0), inlet, boundaries, 3)
u_wall   = DirichletBC(V.sub(0), ((0.0, 0.0)), boundaries, 2)
u_circ   = DirichletBC(V.sub(0), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
u_bar    = DirichletBC(V.sub(0), ((0.0, 0.0)), boundaries, 5) #No slip on geometry in fluid

bcs_u = [u_inlet, u_wall, u_circ, u_bar]

u_inlet_t  = DirichletBC(U, inlet, boundaries, 3)
u_wall_t   = DirichletBC(U, ((0.0, 0.0)), boundaries, 2)
u_circ_t   = DirichletBC(U, ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
u_bar_t    = DirichletBC(U, ((0.0, 0.0)), boundaries, 5) #No slip on geometry in fluid

bcs_u_t = [u_inlet_t, u_wall_t, u_circ_t, u_bar_t]

#Pressure Conditions
p_out = DirichletBC(V.sub(1), 0, boundaries, 4)

bcs_p = [p_out]

#Mesh velocity conditions
w_wall    = DirichletBC(U, ((0.0, 0.0)), boundaries, 2)
w_inlet   = DirichletBC(U, ((0.0, 0.0)), boundaries, 3)
w_outlet  = DirichletBC(U, ((0.0, 0.0)), boundaries, 4)
w_circle  = DirichletBC(U, ((0.0, 0.0)), boundaries, 6)
#w_barwall = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 7)
#w_bar     = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 5)

bcs_w = [w_wall, w_inlet, w_outlet, w_circle]

# Deformation and velocity conditions
v_wall    = DirichletBC(W.sub(0), ((0.0, 0.0)), boundaries, 2)
v_inlet   = DirichletBC(W.sub(0), ((0.0, 0.0)), boundaries, 3)
v_outlet  = DirichletBC(W.sub(0), ((0.0, 0.0)), boundaries, 4)
v_circ    = DirichletBC(W.sub(0), ((0.0, 0.0)), boundaries, 6)
v_barwall = DirichletBC(W.sub(0), ((0.0, 0.0)), boundaries, 7)

d_inlet   = DirichletBC(W.sub(1), ((0, 0)), boundaries, 3)
d_wall    = DirichletBC(W.sub(1), ((0, 0)), boundaries, 2)
d_out     = DirichletBC(W.sub(1), ((0, 0)), boundaries, 4)
d_circ    = DirichletBC(W.sub(1), ((0, 0)), boundaries, 6)
d_barwall = DirichletBC(W.sub(1), ((0, 0)), boundaries, 7)
#bcs_vd = [d_barwall, v_barwall]

bcs_vd = [v_inlet, v_wall, v_circ, v_barwall, \
       d_inlet, d_wall, d_out, d_circ, d_barwall, \
       p_out]

# Deformation conditions TILDE
d_inlet_t   = DirichletBC(U, ((0, 0)), boundaries, 3)
d_wall_t    = DirichletBC(U, ((0, 0)), boundaries, 2)
d_out_t     = DirichletBC(U, ((0, 0)), boundaries, 4)
d_circ_t    = DirichletBC(U, ((0, 0)), boundaries, 6)
d_barwall_t = DirichletBC(U, ((0, 0)), boundaries, 7)

bcs_d_tilde = [d_inlet_t, d_wall_t, d_out_t, d_circ_t, d_barwall_t]


############## Step 0: Extrapolation of the fluid-structure interface

d_tilde = Function(U) #Solution vector of F_expo
#F_expo = inner(u_ - d0 - k*(3./2*v0 - 1./2*v_1 ), phi)*dx_s
#org
F_expo = inner(u_ - d0 - k*(3./2*v0 - 1./2*v_1), phi)*dx

############## Step 1: Definition of new domain

w_next = Function(U)   #Solution w_n+1 of F_smooth
d_move = Function(U)   #Def new domain of Lambda_f

F_smooth = inner(w - 1./k*(d_tilde - d0), phi)*dx_s\
         + inner(grad(w), grad(phi))*dx_f

############## Step 2: ALE-advection-diffusion step (explicit coupling)
I = Identity(2)
def eps(u):
    return 0.5*(grad(u) + grad(u).T)

def F_(U):
    return I + grad(U)

def S(U):
	return (2*mu_s*E(U) + lamda_s*tr(E(U))*I)

def E(U):
	return 0.5*(F_(U).T*F_(U)-I)

def J_(U):
	return det(F_(U))

def P1(U):
	return F_(U)*S(U)

def sigma_f(v,p):
	return 2*mu_f*sym(grad(v)) - p*Identity(2)

def sigma_f_hat(v,p,u):
	return J_(u)*sigma_f(v,p)*inv(F_(u)).T

u0_tilde = Function(U) # Same as u_tilde_n
u_tent = Function(U)   # Tentative velocity: solution of F_tent
#u_last = Function(U)

F_tent = (rho_f/k)*inner(J_(d_tilde)*(u_ - u0), phi)*dx_f \
        + rho_f*inner(J_(d_tilde)*inv(F_(d_tilde))*grad(u_)*(u0_tilde - w_next), phi)*dx_f  \
        + inner(2.*mu_f*J_(d_tilde)*eps(u_)*inv(F_(d_tilde)).T, eps(phi))*dx_f \
        + inner(u_('-') - w('-'), phi('-'))*dS(5)

        #proposed in winterschoolfsi
        #+ inner( 2.*mu_f*J_(d_move)*(grad(u_)*inv(F_(d_move)) + inv(F_(d_move)).T*grad(u_).T )*inv(F_()).T, eps(phi) )*dx_f \

############## Step 3: Projection Step (implicit coupling) ITERATIVE PART

############## Step 3.1: Projection
col_0 = W.sub(0).dofmap().collapse(mesh)[1].values()
vd.vector()[col_0] = w_next.vector()
col_1 = W.sub(1).dofmap().collapse(mesh)[1].values()
vd.vector()[col_1] = d_tilde.vector()


# Pressure update
F_press_upt = (rho_f/k)*inner(J_(d)*(u - u_tent), psi)*dx_f \
- inner(J_(d)*p*inv(F_(d)).T, grad(psi))*dx_f \
+ inner(div(J_(d)*inv(F_(d).T)*u), eta)*dx_f \
+ inner(dot(u('-'),n('-')), eta('-'))*dS(5) \
- 1./k*inner(dot(d('-') - d0('-'), n('-')), eta('-'))*dS(5)
#+ dot(dot(u('-'), n('-')) - 1./k*dot(d('-') - d0('-'), n('-')), psi('-'))*dS(5)


############## Step 3.2: Calculate Solid

u_s, p_s = split(up_sol)
#u_s, p_s = up_sol.split(True)
delta = Constant(1.0E10)
Solid_v = rho_s/k*inner(v - v0, alfa)*dx_s + 0.5*inner(P1(d) + P1(d0), eps(alfa))*dx_s
Solid_d = 1.0/k*inner(d - d0, beta)*dx_s - 0.5*inner(v + v0, beta)*dx_s
#Solid_d = delta*(1.0/k*inner(d - d0, beta)*dx_s - 0.5*inner(v + v0, beta)*dx_s)
Solid_dynamic = inner(P1(d('+'))*n('+'), beta('+'))*dS(5) \
              - inner(sigma_f_hat(u_s('+'), p_s('+'), d('+'))*n('+'), beta('+'))*dS(5)

#Newtonsolver
G = Solid_v + Solid_d + Solid_dynamic

#Reset counters
d_up = TrialFunction(W)
J = derivative(G, vd, d_up)
vd_res = Function(W)

iteration = 0

#Solver parameters
atol, rtol = 1e-7, 1e-7             # abs/rel tolerances
lmbda = 1.0                         # relaxation parameter
residual   = 1                      # residual (To initiate)
rel_res    = residual               # relative residual
max_it    = 15                      # max iterations
Iter = 0

#Step 3

t = 0
T = 15

up_last = Function(V)
u_last, p_last = up_last.split(True)
u_last.assign(u_tent)

vd_last = Function(W)
vd_last.assign(vd)
v_last, d_last = vd_last.split(True)

Re = Um*D/nu
print "SOLVING FOR Re = %f" % Re #0.1 Cylinder diameter

while t < T:

    if t < 2:
        inlet.t = t;
    if t >= 2:
        inlet.t = 2;

    #Step 0:
    #A = assemble(lhs(F_expo), keep_diagonal = True);
    #A.ident_zeros();
    #b = assemble(rhs(F_expo))
    #[bc.apply(A, b) for bc in bcs_d_tilde]
    #solve(A , d_tilde.vector(), b)
    #Org
    solve(lhs(F_expo) == rhs(F_expo), d_tilde, bcs_d_tilde)

    #Do i need black magic here ???
    print "STEP 0: Extrapolation Solved"

    #Step 1:
    solve(lhs(F_smooth) == rhs(F_smooth), w_next, bcs_d_tilde)
    #Project solution to Function vd, to be used as guess for
    #eta_n+1 in step 3.1

    print "STEP 1: Definition New Domain Solved"
    d_move.vector()[:] = d0.vector()[:] + float(k)*w_next.vector()[:]

    col_0 = W.sub(0).dofmap().collapse(mesh)[1].values()
    vd.vector()[col_0] = w_next.vector()

    #Initial guess for d_n+1  step 3.1
    col_1 = W.sub(1).dofmap().collapse(mesh)[1].values()
    vd.vector()[col_1] = d_tilde.vector()

    #Step 2:
    A = assemble(lhs(F_tent), keep_diagonal = True);
    A.ident_zeros();
    b = assemble(rhs(F_tent))

    [bc.apply(A, b) for bc in bcs_u_t]
    solve(A , u_tent.vector(), b)
    u0_tilde.assign(u_tent)
    print "STEP 2: Tentative Velocity Solved"

    eps_f = 1
    eps_s = 1
    iteration = 0

    #Step 3:
    while eps_f > 10E-5 and eps_s > 10E-5 and iteration < 10:
        print "Iteration %d" % iteration
        #Step 3.1:
        A = assemble(lhs(F_press_upt), keep_diagonal = True)
        A.ident_zeros();
        b = assemble(rhs(F_press_upt))
        [bc.apply(A, b) for bc in bcs_u]
        [bc.apply(A, b) for bc in bcs_p]
        solve(A , up_sol.vector(), b)

        u_s, p_s = up_sol.split(True)
        eps_f = errornorm(u_s, u_last, norm_type="l2", degree_rise=2)
        #eps_f = errornorm(p_s, p_last, norm_type="l2", degree_rise=2)

        vd = Newton_manual(G, vd, bcs_vd, J, atol, rtol, max_it, lmbda, vd_res)
        v, d = vd.split(True)
        eps_s = errornorm(d, d_last, norm_type="l2", degree_rise=2)

        u_last.assign(u_s)
        #p_last.assign(p_s)
        d_last.assign(d)

        print "eps_f %.3e: eps_s = %.3e:  iteration = %d " % (eps_f, eps_s, iteration)
        iteration += 1



    up0.assign(up_sol)
    vd1.assign(vd0)
    vd0.assign(vd)

    v0, d0 = vd0.split(True)
    u0, p0 = up0.split(True)
    #u_last.assign(u0)
    Dr = -assemble((sigma_f_hat(u_s, p_s , d)*n)[0]*ds(6))
    Li = -assemble((sigma_f_hat(u_s, p_s , d)*n)[1]*ds(6))
    Dr += -assemble((sigma_f_hat(u_s('-'), p_s('-'), d('-'))*n('-'))[0]*dS(5))
    Li += -assemble((sigma_f_hat(u_s('-'), p_s('-'), d('-'))*n('-'))[1]*dS(5))
    #Drag.append(Dr)
    #Lift.append(Li)

    dsx = d0(coord)[0]
    dsy = d0(coord)[1]
    #dis_x.append(dsx)
    #dis_y.append(dsy)

    if MPI.rank(mpi_comm_world()) == 0:
        print "t = %.4f " %(t)
        print 'Drag/Lift : %g %g' %(Dr,Li)
        print "dis_x/dis_y : %.3e %.3e "%(dsx,dsy)

    print "STEP 3 Solved"
    t += dt
