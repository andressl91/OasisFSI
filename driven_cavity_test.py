from dolfin import *
import numpy as np
set_log_active(False)

#mesh = Mesh("von_karman_street_FSI_fluid.xml")
N = 50
mesh = UnitSquareMesh(N,N)
#plot(mesh,interactive=True)

V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VQ = MixedFunctionSpace([V1,Q])

# BOUNDARIES
Top = AutoSubDomain(lambda x: "on_boundary" and near(x[1],1))
NOS = DomainBoundary()
boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
NOS.mark(boundaries,1)
Top.mark(boundaries,2)
#plot(boundaries,interactive = True)



ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh)


#BOUNDARY CONDITIONS

nos = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 1)
top = DirichletBC(VQ.sub(0), ((1, 0)), boundaries, 2)

bcs = [nos,top]



# TEST TRIAL FUNCTIONS
phi, eta = TestFunctions(VQ)
u,p = TrialFunctions(VQ)

u0 = Function(V1)
u1 = Function(V1)

dt = 0.01
k = Constant(dt)
#EkPa = '62500'
#E = Constant(float(EkPa))

rho_f = 1.0
nu = 0.1
mu_f = rho_f*nu




def sigma_fluid(p,u):
    return -p*Identity(2) + mu_f * (nabla_grad(u) + nabla_grad(u).T)#sym(grad(u))


# Fluid variational form
F = rho_f*((1./k)*inner(u-u1,phi)*dx \
    + inner(dot(u1, grad(u)), phi) * dx) \
    + inner(sigma_fluid(p,u),grad(phi))*dx - inner(div(u),eta)*dx



a = lhs(F)
L = rhs(F)




T = 1.0
t = 0.0
up = Function(VQ)


#A = assemble(a)
counter = 0
while t < T:

    #[bc.apply(A,b) for bc in bcs]
    #solve(A,up.vector(),b)
    solve(a==L,up,bcs)
    u_,p_ = up.split(True)
    print "Timestep: ", t
    #if (counter%100)==0:
    #plot(u_,rescale = True)
    #plot(p_,rescale = True)

    u1.assign(u_)



    #print "Time:",t

    t += dt
psi = Function(Q)
p = TrialFunction(Q)
q = TestFunction(Q)
solve(inner(grad(p), grad(q))*dx == inner(curl(u_), q)*dx, psi, bcs=[DirichletBC(Q, 0, "on_boundary")])
pa = psi.vector().array().argmin()
sort = psi.vector().array().argsort()
#print pa,"--",sort[0], sort[1]
xx = interpolate(Expression("x[0]"), Q)
yy = interpolate(Expression("x[1]"), Q)
xm_1 = xx.vector()[sort[0]]
ym_1 = yy.vector()[sort[0]]
#xm_2 = xx.vector()[sort[-1]]
#ym_2 = yy.vector()[sort[-1]]
print "Center main-eddy: x: %.4f, %.4f " %(xm_1, ym_1)
print "Stream function value at main-eddy: %.4f " %(psi(xm_1,ym_1))
mycurl = project(curl(u_), Q, bcs=[DirichletBC(Q, 0, "on_boundary")])
v_value = mycurl(xm_1,ym_1)
print "Vorticity value at main-eddy: %.4f "%(v_value)

plot(u_,interactive=True)
