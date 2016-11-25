from mixed_domain_FSI import *
from numpy import zeros, where
import numpy as np
import sys


#parameters['allow_extrapolation']=True
parameters["std_out_all_processes"] = False
parameters["linear_algebra_backend"] = "PETSc"
set_log_active(False)
set_log_level(PROGRESS)



vfile = File("RESULTS/Hron_Lagrange/v.pvd") # xdmf
pfile = File("RESULTS/Hron_Lagrange/p.pvd")
ufile = File("RESULTS/Hron_Lagrange/u.pvd")


mesh = Mesh('fluid_new.xml')
#mesh = refine(mesh)
#mesh = refine(mesh)
LEN = len(mesh.coordinates())
Solid_mesh = Elastic()
SD = MeshFunction('size_t', mesh, mesh.topology().dim())
SD.set_all(0)
Solid_mesh.mark(SD,1)

# DEFINING BOUNDARIES
boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
inlet.mark(boundaries,1)
top.mark(boundaries,2)
bottom.mark(boundaries,3)
outlet.mark(boundaries,4)
circle.mark(boundaries,6)
interface.mark(boundaries,5)
Elastic_to_Circle().mark(boundaries,10)
Solid_only().mark(boundaries,8)
inner_solid_circle.mark(boundaries,9)

File('bnd.pvd') << boundaries



dt = 0.001   # use 0.0003 for oscillations
T = 1
# TEST AND TRIALFUNCTIONS
V = VectorFunctionSpace(mesh,'CG',1)
P = FunctionSpace(mesh,'CG',1)
U = VectorFunctionSpace(mesh,'CG', 1)
VPU = MixedFunctionSpace([V,P,U])
print 'dofs: ', VPU.dim()
vpu = TrialFunction(VPU)
phietapsi = TestFunction(VPU)

v,p,u = split(vpu)
phi,eta,psi = split(phietapsi)



# PHYSICAL PARAMETERS
h = mesh.hmin()

# FLUID
FSI = 1
nu = 10**-3
rho_f = 1.0*1e3
mu_f = rho_f*nu
U_in = [0.2, 1.0, 2.0][FSI-1]   # Reynolds vel

# SOLID
Pr = 0.4
mu_s = [0.5, 0.5, 2.0][FSI-1]*1e6
rho_s = [1.0, 10, 1.0][FSI-1]*1e3
lamda = 2*mu_s*Pr/(1-2.*Pr)

I = Identity(2)

def Eij(U):
	return sym(grad(U))# - 0.5*dot(grad(U),grad(U))

def F(U):
	return (I + grad(U))

def J(U):
	return det(F(U))

def E(U):
	return 0.5*(F(U).T*F(U)-I)

def S(U):
	return (2*mu_s*E(U) + lamda*tr(E(U))*I)

def P1(U):
	return F(U)*S(U)

def sigma_f(v,p):
	return 2*mu_f*sym(grad(v)) - p*Identity(2)

def sigma_s(u):
	return 2*mu_s*sym(grad(u)) + lamda*tr(sym(grad(u)))*I

def sigma_f_hat(v,p,u):
	return J(u)*sigma_f(v,p)*inv(F(u)).T




# INITIAL AND BOUNDARY CONDITIONS

# FLUID

class v_in(Expression):
	def __init__(self):
		self.t = 0
	def eval(self,value,x):
		value[0] = 0.5*(1-np.cos(self.t*np.pi/2))*1.5*U_in*x[1]*(H-x[1])/((H/2.0)**2)
		value[1] = 0
	def value_shape(self):
		return (2,)


noslip = Constant((0.0,0.0))
vf = v_in()#Expression(('0.5*(1-cos(t*pi/2))*1.5*U*x[1]*(H-x[1])/pow((H/2.0),2)','0.0'),H=H,U=U_,t=0)

bcv0 = DirichletBC(VPU.sub(0),noslip,boundaries,1)     # inlet
bcv1 = DirichletBC(VPU.sub(0),vf,boundaries,1)     # inlet
bcv3 = DirichletBC(VPU.sub(0),noslip,boundaries,3) # bottom
bcv5 = DirichletBC(VPU.sub(0),noslip,boundaries,5) # interface
bcv2 = DirichletBC(VPU.sub(0),noslip,boundaries,2) # Top
bcv6 = DirichletBC(VPU.sub(0),noslip,boundaries,6) # Circle
bcv9 = DirichletBC(VPU.sub(0),noslip,boundaries,9)
bcv10 = DirichletBC(VPU.sub(0),noslip,boundaries,10) # Circle to elastic


bcv = [bcv1, bcv3, bcv2, bcv6, bcv9, bcv10]
bcdv = [bcv0, bcv3, bcv2, bcv6, bcv9, bcv10]

# SOLID

# MESH DISPLACEMENT

bcu1 = DirichletBC(VPU.sub(2),noslip,boundaries,1)  # inlet
bcu4 = DirichletBC(VPU.sub(2),noslip,boundaries,4)  # outlet
bcu2 = DirichletBC(VPU.sub(2),noslip,boundaries,2)  # top
bcu3 = DirichletBC(VPU.sub(2),noslip,boundaries,3)  # bottom
bcu5 = DirichletBC(VPU.sub(2),noslip,boundaries,5)  # interface
bcu6 = DirichletBC(VPU.sub(2),noslip,boundaries,6)  # circle
bcu9 = DirichletBC(VPU.sub(2),noslip,boundaries,9)
bcu10 = DirichletBC(VPU.sub(2),noslip,boundaries,10) # circle to elastic
bcu = [bcu1,bcu2,bcu3,bcu4,bcu9,bcu10,bcu6]

bcp1 = DirichletBC(VPU.sub(1),Constant(0),boundaries,8)
bcp2 = DirichletBC(VPU.sub(1),Constant(0),boundaries,9)

bcp = [bcp1,bcp2]



bcs = bcv+bcu+bcp
dbcs = bcdv+bcu

# CREATE FUNCTIONS
v0 = Function(V)

v1 = Function(V)#,'initial_data/u.xml')
u1 = Function(U)

VPU_ = Function(VPU)
VPU1 = Function(VPU)

# Define coefficients
k = Constant(dt)
f = Constant((0, 0))
n = FacetNormal(mesh)
g = Constant((0,2.0))



dS = Measure('dS')[boundaries]
dx = Measure('dx')[SD]
ds = Measure('ds')[boundaries]

#q_degree = 3
#dx = dx(metadata={'quadrature_degree': q_degree})

#print assemble(n[0]*ds(6)), 'negative if n out of circle'
#print assemble(n('-')[0]*dS(5)), 'positive if n(-) out of solid'
#sys.exit()
# n('+') out of solid, n('-') into solid
# n into circle

epsilon = 1e10


dx_s = dx(1,subdomain_data=SD)
dx_f = dx(0,subdomain_data=SD)

un = 0.5*(u+u1)
vn = 0.5*(v+v1)

FS = rho_s/k*inner(J(u)*(v-v1),phi)*dx_s + inner(P1(u),grad(phi))*dx_s + epsilon*1/k*inner(u-u1,psi)*dx_s - epsilon*inner(v,psi)*dx_s

FS_L = rho_s/k*inner(J(u1)*(v-v1),phi)*dx_s + inner(F(u1)*sigma_s(u),grad(phi))*dx_s + 1/k*inner(u-u1,psi)*dx_s - 0.5*inner(v+v1,psi)*dx_s


# FLUID
factor = Constant(7e3)
O_ = factor*dot(u('-'),u('-'))
O = factor*dot(u,u)+1


FF =rho_f/k*inner(J(u)*(v-v1),phi)*dx_f + rho_f*inner(J(u)*inv(F(u))*grad(v)*(v-1./k*(u-u1)),phi)*dx_f + inner(sigma_f_hat(v,p,u), grad(phi))*dx_f - inner(eta,div(J(u)*inv(F(u).T)*v))*dx_f + 1./k*inner(u-u1,psi)*dx_f + inner(grad(u),grad(psi))*dx_f - 0.05*h**2*inner(grad(p),grad(eta))*dx_f

FF_L = rho_f/k*inner(J(u1)*(v-v1),phi)*dx_f + rho_f*inner(J(u1)*grad(v1)*inv(F(u1))*(v-1./k*(u-u1)),phi)*dx_f + inner(sigma_f_hat(v,p,u1), grad(phi))*dx_f - inner(eta,div(J(u1)*inv(F(u1).T)*v))*dx_f + inner(grad(u),grad(psi))*dx_f - 0.1*h**2*inner(grad(p),grad(eta))*dx_f


t = dt

xA = where(mesh.coordinates()[:,0] == 0.6)
yA = where(mesh.coordinates()[:,1] == 0.2)

try:
	xA = where(abs(mesh.coordinates()[:,0] - 0.6) < h)
	yA = where(abs(mesh.coordinates()[:,1] - 0.2) < h)
	for idx in xA[0]:
		if idx in yA[0]:
			coord = idx
	xA = float(mesh.coordinates()[coord,0])
	yA = float(mesh.coordinates()[coord,1])
except:
	coord = 0


count = 0
while t < T + DOLFIN_EPS:# and (abs(FdC) > 1e-3 or abs(FlC) > 1e-3):
	if t <= 1:
		vf.t=2*t
	else:
		vf.t = 2.0
	VPU_k = Function(VPU)
	F_L = FF_L + FS_L
	solve(lhs(F_L)==rhs(F_L),VPU_k,bcs)
	v_k,p_k,w_k = VPU_k.split(True)
	if t == dt:
		plot(v_k)
		interactive()
	D = action(FF+FS,VPU_k)
	Jk = derivative(D,VPU_k,vpu)
	dVPU = Function(VPU)
	eps = 1.0
	max_iter = 7
	k_iter = 0
	while eps > 1E-6 and k_iter < max_iter:
		D = action(FF+FS,VPU_k)
		Jk = derivative(D, VPU_k, vpu)
		A = assemble(Jk)
		b = assemble(-D)
		A.ident_zeros()
		[dbc.apply(A,b,dVPU.vector()) for dbc in dbcs]
		solve(A,dVPU.vector(),b)
		VPU_.vector()[:] = VPU_k.vector()[:] + dVPU.vector()[:]
		eps = norm(dVPU.vector())
		print 'k: ',k_iter, 'error: %.3e' %eps
		k_iter += 1
		VPU_k.assign(VPU_)
		#b = assemble(L)

	v_,p_,u_ = VPU_.split(True)
	Dr = -assemble((sigma_f_hat(v_,p_,u_)*n)[0]*ds(6))
	Li = -assemble((sigma_f_hat(v_,p_,u_)*n)[1]*ds(6))
	print 't=%.4f Drag/Lift on circle: %g %g' %(t,Dr,Li)



	Dr += -assemble((sigma_f_hat(v_('-'),p_('-'),u_('-'))*n('-'))[0]*dS(5))
	Li += -assemble((sigma_f_hat(v_('-'),p_('-'),u_('-'))*n('-'))[1]*dS(5))
	YD = mesh.coordinates()[coord,1]-yA
	XD = mesh.coordinates()[coord,0]-xA
	if coord != 0:
		print '%g %g' %(Dr,Li)#,XD,YD)
	if count%1==0:
		vfile << v_
		pfile << p_
		ufile << u_
	v1.assign(v_)
	u1.assign(u_)


	t += dt

	count += 1
