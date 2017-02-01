from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
# Mesh
import argparse
from argparse import RawTextHelpFormatter
def parse():
    parser = argparse.ArgumentParser(description="Implementation of Turek test case FSI\n"
    "For details: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.550.1689&rep=rep1&type=pdf",\
     formatter_class=RawTextHelpFormatter, \
     epilog="############################################################################\n"
     "Example --> python ALE_PM.py \n"
     "Example --> python ALE_PM.py -v_deg 2 -p_deg 1 -w_deg 2 -dt 0.5 -T 10 -step 1   (Refines mesh one time, -rr for two etc.) \n"
     "############################################################################")
    group = parser.add_argument_group('Parameters')
    group.add_argument("-p_deg",       type=int,   help="Set degree of pressure                     --> Default=1", default=1)
    group.add_argument("-v_deg",       type=int,   help="Set degree of velocity                     --> Default=2", default=2)
    group.add_argument("-w_deg",       type=int,   help="Set degree of displacement                 --> Default=2", default=2)
    group.add_argument("-T",           type=float, help="End time                     --> Default=20", default=20)
    group.add_argument("-dt",          type=float, help="Time step                     --> Default=0.5", default=0.5)
    group.add_argument("-step",          type=float, help="savestep                     --> Default=1", default=1)
    group.add_argument("-r", "--refiner", action="count", help="Mesh-refiner using built-in FEniCS method refine(Mesh)")
    group.add_argument("-beta",          type=float, help="AC factor                     --> Default=0.5", default=0.01)
    group.add_argument("-test",       type=int,   help="Implementation                 --> Default=1", default=1)
    return parser.parse_args()
args = parse()
v_deg = args.v_deg
p_deg = args.p_deg
w_deg = args.w_deg
T = args.T
dt = args.dt
beta = args.beta
step = args.step
test = args.test
mesh = RectangleMesh(Point(0,0), Point(2, 1), 20, 20, "crossed")

# FunctionSpaces
V1 = VectorFunctionSpace(mesh, "CG", v_deg) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", w_deg) # fluid displacement velocity
Q = FunctionSpace(mesh, "CG", p_deg)
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
Wm = 0.1
#inlet = Expression((("Wm","0")),Wm = Wm,t=0)
inlet = Expression((("cos(t)*(x[0]-2)","0")),Wm = Wm,t=0)


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

k = Constant(dt)

# Fluid properties
rho_f   = Constant(1.0)
nu_f = Constant(1.0)
mu_f    = Constant(1.0)
I = Identity(2)
def Eij(U):
	return sym(grad(U))# - 0.5*dot(grad(U),grad(U))

def F_(U):
	return (I + grad(U))

def J_(U):
	return det(F_(U))

def E(U):
	return 0.5*(F_(U).T*F_(U)-I)

def S(U):
	return (2*mu_s*E(U) + lamda_s*tr(E(U))*I)

def P1(U):
	return F_(U)*S(U)

def sigma_f(v,p):
	return 2*mu_f*sym(grad(v)) - p*Identity(2)

def sigma_s(u):
	return 2*mu_s*sym(grad(u)) + lamda_s*tr(sym(grad(u)))*I

def sigma_f_hat(v,p,u):
	return J_(u)*sigma_f(v,p)*inv(F_(u)).T
def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)
def sigma_f_new(u,p,d):
	return -p*I + mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

d = k*w_

# Fluid variational form
if test == 1: # The richter way
    F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
    F_fluid += rho_f*inner(J_(d)*inv(F_(d))*dot((u - w_),grad(u)), phi)*dx
    F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
    F_fluid += inner(J_(d)*sigma_f_new(u,p,d)*inv(F_(d)).T, grad(phi))*dx

if test == 2: # The richter way with tensor written out
    F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
    F_fluid += rho_f*inner(J_(d)*inv(F_(d))*dot((u - w_),grad(u)), phi)*dx
    F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
    F_fluid += inner(J_(d)*2*mu_f*epsilon(u)*inv(F_(d))*inv(F_(d)).T ,epsilon(phi) )*dx
    F_fluid -= inner(J_(d)*p*I*inv(F_(d)).T, grad(phi))*dx

if test == 3: # Written
    F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
    F_fluid += rho_f*inner(J_(d)*grad(u)*inv(F_(d))*(u - w_), phi)*dx
    F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
    F_fluid += inner(J_(d)*2*mu_f*epsilon(u)*inv(F_(d))*inv(F_(d)).T ,epsilon(phi) )*dx
    F_fluid -= inner(J_(d)*p*I*inv(F_(d)).T, grad(phi))*dx

if test == 4: # Written with tensor and grad on the left side.
    F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
    F_fluid += rho_f*inner(J_(d)*grad(u)*inv(F_(d))*(u - w_), phi)*dx
    F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
    F_fluid += inner(J_(d)*sigma_f_new(u,p,d)*inv(F_(d)).T, grad(phi))*dx



# laplace d = 0
F2 =  k*(inner(grad(w), grad(psi))*dx - inner(grad(w)*n, psi)*ds)

t = 0.0

u_file = XDMFFile(mpi_comm_world(), "new_results/velocity.xdmf")
d_file = XDMFFile(mpi_comm_world(), "new_results/d.xdmf")
w_file = XDMFFile(mpi_comm_world(), "new_results/w.xdmf")
p_file = XDMFFile(mpi_comm_world(), "new_results/pressure.xdmf")

for tmp_t in [u_file, d_file, p_file, w_file]:
    tmp_t.parameters["flush_output"] = True
    tmp_t.parameters["multi_file"] = 1
    tmp_t.parameters["rewrite_function_mesh"] = False
#w_.vector()[:] *= float(k) # gives displacement to be used in ALE.move(w_)

#plot(w_,interactive=True)
time_array = []
flux = []
while t <= T:
    print "Time: ",t
    inlet.t = t
    time_array.append(t)
    solve(lhs(F2)==rhs(F2), w_, bcs_w)
    u_bc.init(w_)

    solve(F_fluid==0, up_, bcs_u,solver_parameters={"newton_solver": \
    {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,\
    "maximum_iterations":100,"relaxation_parameter":1.0}})
    u,p = up_.split(True)
    #print "u-w : ", assemble(dot(u-w_,n)*ds(4))
    #plot(d,mode = "displacement")
    plot(u)
    #flux.append(assemble(J_*dot(u,n)*ds(3)))
    u0.assign(u)
    u_file << u
    #d_file << d

    #p_file << p
    #w_file << w_

    # To see the mesh move with a give initial w
    #ALE.move(mesh,w_)
    #plot(mesh)#,interactive = True)

    t += dt
#print len(flux),len(time_array)
#plt.plot(time_array,flux);plt.title("Flux, with N = 20"); plt.ylabel("Flux out");plt.xlabel("Time");plt.grid();
#plt.show()
