from dolfin import *
import numpy as np
set_log_active(False)

mesh = Mesh("von_karman_street_FSI_structure.xml")
#print mesh.coordinates()
for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001):
        print coord[0], coord[1]
#N = 10
#mesh = UnitSquareMesh(N,N)
#plot(mesh,interactive=True)

#V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V = VectorFunctionSpace(mesh, "CG", 1) # Mesh movement
#Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

#VVQ = MixedFunctionSpace([V1,V2,Q])

print "Dofs: ",V.dim(), "Cells:", mesh.num_cells()

Left  = AutoSubDomain(lambda x: "on_boundary" and (near(x[0], 0.6)))
"""class Left(SubDomain):
	def inside(self,x,on_boundary):
		return on_boundary and not(near(x[0],0.6) or near(x[1],0.19) or near(x[1],0.21))
"""
boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Left.mark(boundaries,1)
plot(boundaries,interactive=True)
bc1 = DirichletBC(V, ((0,0)),boundaries, 1)
bcs = [bc1]

"""Pr = 0.479
rho_s = Constant(1.75*rho_f)
lamda = Constant(E*Pr/((1.0+Pr)*(1.0-2*Pr)))
mu_s = Constant(E/(2*(1.0+Pr)))"""

rho_s = 10.0e3
mu_s = 0.5*10.0e6
nu_s = 0.4
E = 1.4*10.0e6
lamda = Constant(E*nu_s/((1.0+nu_s)*(1.0-2*nu_s)))
g = Constant((0,-2*rho_s))



def sigma_structure(d):
    return 2*mu_s*sym(grad(d)) + lamda*tr(sym(grad(d)))*Identity(2)

# TEST TRIAL FUNCTIONS
psi = TestFunction(V)
w = TrialFunction(V)

#w = Function(V)
w0 = Function(V)
U1 = Function(V)

dt = 0.1
k = Constant(dt)

#Structure Variational form
U = U1 + k*w

G = rho_s*((1./k)*inner(w-w0,psi))*dx +rho_s*inner(dot(grad(w0),w),psi)*dx \
+ inner(sigma_structure(U),grad(psi))*dx \
- inner(g,psi)*dx

a = lhs(G)
L = rhs(G)

A = assemble(a)
counter = 0
t = 0
T = 10.0
w_ = Function(V)
while t < T:
    print "Time: ",t
    b = assemble(L)
    #A.ident_zeros()
    #[bc.apply(A,b) for bc in bcs]
    #solve(A,w_.vector(),b)
    solve(a==L,w_,bcs)
    t += dt
    w0.assign(w_)
    print w_(coord)

    plot(w_,mode="displacement")#, interactive=True)
