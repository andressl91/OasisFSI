from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

#from parser import *
#from mappings import *
"""from Hron_Turek import *
from variational_form import *
from Traction import *"""
dt = 0.05
I = Identity(2)
def F_(U):
	return (I + grad(U))

def J_(U):
	return det(F_(U))

mesh = RectangleMesh(Point(0,0), Point(2, 1), 20, 20, "crossed")
V1 = VectorFunctionSpace(mesh, "CG", 1) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # displacement
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VQ = MixedFunctionSpace([V1,Q])
#time0 = time()
#parameters["num_threads"] = 2
parameters["allow_extrapolation"] = True

u_file = XDMFFile(mpi_comm_world(), "PM_RN"+"/dt-"+str(dt)+"/velocity.xdmf")
w_file = XDMFFile(mpi_comm_world(), "PM_RN"+"/dt-"+str(dt)+"/w.xdmf")
df_file = XDMFFile(mpi_comm_world(),"PM_RN"+"/dt-"+str(dt)+"/df.xdmf")
#p_file = XDMFFile(mpi_comm_world(), "PM_RN"+"/dt-"+str(dt)+"/pressure.xdmf")

for tmp_t in [u_file,w_file,df_file]:  # d_file, p_file]:
    tmp_t.parameters["flush_output"] = True
    tmp_t.parameters["multi_file"] = 0
    tmp_t.parameters["rewrite_function_mesh"] = False

#print "Dofs: ",VQ.dim(), "Cells:", mesh.num_cells()

nu = 10**-3
rho_f = 1.0*1e3
mu_f = rho_f*nu

# SOLID
Pr = 0.4
mu_s = 0.51e6
rho_s = 1.01e3
lamda_s = 2*mu_s*Pr/(1-2.*Pr)

def sigma_dev(U): #linear solid stress tensor
	return 2*mu_s*sym(grad(U)) + lamda_s*tr(sym(grad(U)))*Identity(2)

def sigma_f_new(u,p,d):
	return -p*I + mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

# TEST TRIAL FUNCTIONS
phi, gamma = TestFunctions(VQ)
psi = TestFunction(V1)
u, p  = TrialFunctions(VQ)

w = TrialFunction(V1)

up_ = Function(VQ)
u_, p_  = up_.split()

uf = Function(V1)
pf = Function(Q)

up0 = Function(VQ)
u0, p0 = up0.split()
up_res = Function(VQ)

#XI = TestFunction(V1)
df = Function(V1)
df0 = Function(V1)

#df, _ = df_res.split(True)
d = TrialFunction(V1)
d__ = Function(VQ)
d_, _ = d__.split(True)
d0 = Function(V1)
d1 = Function(V1)
d2 = Function(V1)
d_res = Function(V1)
w_ = Function(V1)

k = Constant(dt)
I = Identity(2)
alpha = 1.0
beta = 0.01
#h =  mesh.hmin()

time_list = []

#cos(t)*
inlet = Expression((("0.2", "0")),t=0)
#inlet = Expression((("2","0")),t=0)
h = mesh.hmin()


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

u_bc = U_bc(degree=2)

u_wall   = DirichletBC(VQ.sub(0), u_bc, boundaries, 4)
u_inlet   = DirichletBC(VQ.sub(0), inlet, boundaries, 2)

# Mesh velocity conditions
w_inlet   = DirichletBC(V1, inlet, boundaries, 2)
w_outlet  = DirichletBC(V1, ((0.0, 0.0)), boundaries, 3)

# Pressure Conditions
p_out = DirichletBC(VQ.sub(1), 0, boundaries, 3)

#Assemble boundary conditions
bcs_w = [w_inlet, w_outlet]
bcs_u = [u_wall, p_out]

f = Function(V1)
#f, _ = f_.split()
#F_Ext = inner(grad(d), grad(psi))*dx_f - inner(f, psi)*dx_f #- inner(grad(d)*n, psi)*ds
F_Ext =  k*(inner(grad(w), grad(psi))*dx - inner(grad(w)*n, psi)*ds) - inner(f, psi)*dx

df = k*w_
#w_ = df/k
delta = 1E10
F_fluid = (rho_f/k)*inner(J_(df)*(u - u0), phi)*dx
F_fluid += rho_f*inner(J_(df)*grad(u0)*inv(F_(df))*(u - w_), phi)*dx
F_fluid += inner(J_(df)*sigma_f_new(u,p,df)*inv(F_(df)).T, grad(phi))*dx
F_fluid -= inner(div(J_(df)*inv(F_(df)).T*u), gamma)*dx
F_fluid += inner(J_(df)*sigma_f_new(u,p,df)*inv(F_(df)).T*n, phi)*ds(2)
F_fluid += inner(sigma_dev(df)*n, phi)*ds(2)
F_fluid -= beta*h*h*inner(J_(df)*inv(F_(df).T)*grad(p), grad(gamma))*dx
#F_fluid += inner(u, phi)*dx
#F_fluid -= inner(w_, phi)*dx
F_fluid += delta*(inner(df/k,phi)*ds(2) - inner(u,phi)*ds(2))



t = dt

# Newton parameters
atol = 1e-6;rtol = 1e-6; max_it = 100; lmbda = 1.0;

ones_d = Function(V1)
ones_u = Function(VQ)
ones_d.vector()[:] = 1.
ones_u.vector()[:] = 1.

adf = lhs(F_Ext)
Ldf = rhs(F_Ext)

af = lhs(F_fluid)
bf = rhs(F_fluid)
T = 10
while t <= T:
    print "Time t = %.5f" % t

    inlet.t = t

    Adf = assemble(adf, keep_diagonal=True)
    Adf.ident_zeros()
    Bdf = assemble(Ldf)

    [bc.apply(Adf, Bdf) for bc in bcs_w]

    solve(Adf, w_.vector(), Bdf)

    u_bc.init(w_)
    #df = k*w_

    Mass_s_b = assemble(inner(d_, phi)*ds(2))
    #Mass_u_b = assemble(delta*inner(u, phi)*ds(2))

    #Mass_s_b_rhs = assemble((rho_s/k)*inner((2*((d0-d1)/k) - ((d1 - d2)/k)), phi)*ds(2))
    #Mass_s_b_rhs = assemble(delta*(rho_s/k)*inner(w_, phi)*ds(2))

    Mass_s_b_L = Mass_s_b*ones_u.vector() #Mass structure matrix lumped
    #Mass_u_b_L = Mass_u_b*ones_u.vector() #Mass structure matrix lumped

    #Mass_u_b_L = Mass_s_b_L*Mass_u_b_L
    #Mass_s_and_rhs = Mass_s_b_L*Mass_s_b_rhs

    mass_form = inner(u,phi)*dx
    M_lumped = assemble(mass_form)
    M_lumped.zero()
    M_lumped.set_diagonal(Mass_s_b_L)

    A = assemble(af, keep_diagonal=True)#, tensor=A) #+ Mass_s_b_L
    A += M_lumped

    A.ident_zeros()

    B = assemble(bf)
    #B -= Mass_s_and_rhs

    [bc.apply(A, B) for bc in bcs_u]

    solve(A, up_.vector(), B)

    #up0.assign(up_)
    u_, p_ = up_.split(True)
    u0.assign(u_); p0.assign(p_)

    u_file << u_
    df_file << d0
    w_file << w_

    #p_file << p_

    #d2.assign(d1)
    d1.assign(d0)
    d0.assign(df)
    #df0.assign(df)
    #plot(d0, mode="displacement")#,interactive=True)
    plot(u_)
    t += dt
