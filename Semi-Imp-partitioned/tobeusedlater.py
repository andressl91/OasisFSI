from dolfin import *
from solvers import Newton_manual
from variationalforms import *
from Utils.argpar import *

#print sa

parameters['allow_extrapolation']=True

############# Get Problem specifics
args = parse()
problem = args.problem
exec("from Problems.%(problem)s import *" % vars())


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

############## Step 0: Extrapolation of the fluid-structure interface

d_tilde = Function(U) #Solution vector of F_expo
#d_tilde = step0(d_tilde, d0, v0, v_1, k)


############## Step 1: Definition of new domain

w_next = Function(U)   #Solution w_n+1 of F_smooth
d_move = Function(U)   #Def new domain of Lambda_f
#F_smooth = step1(w, k, phi, dx_f)


############## Step 2: ALE-advection-diffusion step (explicit coupling)

u0_tilde = Function(U) # Same as u_tilde_n
u_tent = Function(U)   # Tentative velocity: solution of F_tent
#u_last = Function(U)
F_tent = step2(d_move, w_next, u_, u0, u0_tilde, phi, dx_f, mu_f, rho_f, k)


############## Step 3: Projection Step (implicit coupling) ITERATIVE PART

############## Step 3.1: Projection

#col_0 = W.sub(0).dofmap().collapse(mesh)[1].values()
#vd.vector()[col_0] = w_next.vector()
#col_1 = W.sub(1).dofmap().collapse(mesh)[1].values()
#vd.vector()[col_1] = d_tilde.vector()

F_press_upt = step3_1(d, d0, u, u_tent, p, psi, eta, n, dx_f, rho_f, k, dS)


############## Step 3.2: Calculate Solid

u_s, p_s = split(up_sol)
#u_s, p_s = up_sol.split(True)
F_s = step3_2(v, v0, d, d0, u_s, p_s, k, mu_f, mu_s, rho_s, lamda_s, n, dx_s, alfa, beta, dS)


############## Define BCS ##################

class U_bc(Expression):
    def __init__(self, d_tilde, d0, k):
        self.d_tilde = d_tilde
        self.d0 = d0
        self.k = k
    def eval(self,value,x):
        #x_value, y_value = self.w.vector()[[x[0], x[1]]]
        value[0], value[1] = 1./self.k*(self.d_tilde(x) - self.d0(x))
        #value[0] = x_value
        #value[1] = y_value
    def value_shape(self):
        return (2,)


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

w_bc_bar = U_bc(d_tilde, d_tilde, k)
# Deformation conditions TILDE
w_inlet   = DirichletBC(U, ((0, 0)), boundaries, 3)
w_wall    = DirichletBC(U, ((0, 0)), boundaries, 2)
w_out     = DirichletBC(U, ((0, 0)), boundaries, 4)
w_circ    = DirichletBC(U, ((0, 0)), boundaries, 6)
w_barwall = DirichletBC(U, ((0, 0)), boundaries, 7)
w_bar     = DirichletBC(U, w_bc_bar, boundaries, 5)

bcs_w = [w_inlet, w_wall, w_out, w_circ, w_barwall, w_bar]


#Reset counters
d_up = TrialFunction(W)
J = derivative(F_s, vd, d_up)
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
    d_tilde = step0(d_tilde, d0, v0, v_1, k)

    print "STEP 0: Extrapolation Solved"

    #Step 1:

    #w_vector()[:] = d0.vector()[:] + float(k)*w_next.vector()[:]

    F_smooth = step1(w, k, phi, dx_f, dx_s)
    A = assemble(lhs(F_smooth), keep_diagonal = True)
    A.ident_zeros()
    b = assemble(rhs(F_smooth))
    [bc.apply(A, b) for bc in bcs_w]
    solve(A, w_next.vector(), b)
    #solve(lhs(F_smooth) == rhs(F_smooth), w_next, bcs_w)
    #Project solution to Function vd, to be used as guess for
    #eta_n+1 in step 3.1

    print "STEP 1: Definition New Domain Solved"
    d_move.vector()[:] = d0.vector()[:] + float(k)*w_next.vector()[:]

    col_0 = W.sub(0).dofmap().collapse(mesh)[1].values()
    vd.vector()[col_0] = w_next.vector()

    #Initial guess for d_n+1  step 3.1
    col_1 = W.sub(1).dofmap().collapse(mesh)[1].values()
    vd.vector()[col_1] = d_move.vector()

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
    while eps_f > 10E-4 or eps_s > 10E-4 and iteration < 10:
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

        vd = Newton_manual(F_s, vd, bcs_vd, J, atol, rtol, max_it, lmbda, vd_res)
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
    Dr = -assemble((sigma_f_hat(u_s, p_s , d, mu_f)*n)[0]*ds(6))
    Li = -assemble((sigma_f_hat(u_s, p_s , d, mu_f)*n)[1]*ds(6))
    Dr += -assemble((sigma_f_hat(u_s('-'), p_s('-'), d('-'), mu_f)*n('-'))[0]*dS(5))
    Li += -assemble((sigma_f_hat(u_s('-'), p_s('-'), d('-'), mu_f)*n('-'))[1]*dS(5))
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
