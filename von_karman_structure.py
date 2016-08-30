from dolfin import *
import numpy as np
set_log_active(False)

mesh = Mesh("von_karman_street_FSI_structure.xml")
#print mesh.coordinates()
for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001):
        print coord
        break
#N = 10
#mesh = UnitSquareMesh(N,N)
#plot(mesh,interactive=True)

#V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V = VectorFunctionSpace(mesh, "CG", 1) # Mesh movement
#Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

#VVQ = MixedFunctionSpace([V1,V2,Q])

print "Dofs: ",V.dim(), "Cells:", mesh.num_cells()


BarLeftSide =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
BarLeftSide.mark(boundaries,1)
plot(boundaries,interactive=True)
bc1 = DirichletBC(V, ((0,0)),boundaries, 1)
bcs = [bc1]

"""Pr = 0.479
rho_s = Constant(1.75*rho_f)
lamda = Constant(E*Pr/((1.0+Pr)*(1.0-2*Pr)))
mu_s = Constant(E/(2*(1.0+Pr)))"""

rho_s = 1.0E3
mu_s = 0.5E6
nu_s = 0.4
E = 1.4E6
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

dt = 0.01
k = Constant(dt)

#Structure Variational form
U = U1 + k*w

G = rho_s*((1./k)*inner(w-w0,psi))*dx + rho_s*inner(dot(grad(w0),w),psi)*dx \
+ inner(sigma_structure(U),grad(psi))*dx \
- inner(g,psi)*dx
#rho_s*((1./k)*inner(w-w0,psi))*dx
a = lhs(G)
L = rhs(G)

A = assemble(a)
counter = 0
t = 0
T = 10.0
w_ = Function(V)
"""solve(a==L,w_,bcs)
w_.vector()[:] *= float(k)
U1.vector()[:] += w_.vector()[:]
print (U1(coord)), coord
plot(U1,interactive=True)"""


while t < T:
    print "Time: ",t
    b = assemble(L)
    #A.ident_zeros()
    [bc.apply(A,b) for bc in bcs]
    solve(A,w_.vector(),b)
    #solve(a==L,w_,bcs)
    t += dt
    w0.assign(w_)
    w_.vector()[:] *= float(k)
    U1.vector()[:] += w_.vector()[:]
    #

    plot(U1,mode="displacement")#, interactive=True)
    print coord
    print "U1: ", U1(coord)
    print "w: ", w_(coord)
