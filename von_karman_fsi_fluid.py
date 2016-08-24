from dolfin import *

mesh = Mesh("von_karman_street_FSI_fluid.xml")
#plot(mesh,interactive=True)

V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # Mesh movement
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VVQ = MixedFunctionSpace([V1,V2,Q])

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
test = DomainBoundary()

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
test.mark(boundaries, 1)
Inlet.mark(boundaries, 2)
Outlet.mark(boundaries, 3)


ds = Measure("ds", subdomain_data = boundaries)
n = FacetNormal(mesh)
#plot(boundaries,interactive=True)

Um = 1.5
H = 0.41
inlet = Expression(("1.5*Um*4.0*x[1]*(H-x[1])/pow(H/2.0,2)","0"),t=0.0,Um = Um,H=H)
u_inlet = DirichletBC(VVQ.sub(0), (inlet), boundaries, 2)
nos = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 1)
bcs = [nos,u_inlet]

phi, psi, eta = TestFunctions(VVQ)
u,w,p = TrialFunctions(VVQ)

u0 = Function(V1)
u1 = Function(V1)
w0 = Function(V2)
U1 = Function(V2)

dt = 0.01
k = Constant(dt)
#EkPa = '62500'
#E = Constant(float(EkPa))
"""
rho_f = Constant(1./1000)		# [g/mm]
nu_f = Constant(0.658)			# [mm**2/s]
mu_f = Constant(nu_f*rho_f)		# [g/(mm*s)]
Pr = 0.479
rho_s = Constant(1.75*rho_f)
lamda = Constant(E*Pr/((1.0+Pr)*(1.0-2*Pr)))
mu_s = Constant(E/(2*(1.0+Pr)))
lamda = Constant("0.0105e9")"""
rho_f = 100.0
mu_f = 1.0
lamda = Constant("0.0105e9")
mu_s = Constant("0.00230e9")
rho_s =Constant(1.45e3)


def sigma_structure(d):
    return 2*mu_s*sym(grad(d)) + lamda*tr(sym(grad(d)))*Identity(2)

def sigma_fluid(p,u):
    return -p*Identity(2) + 2*mu_f * sym(grad(u))

# Fluid variational form
F = rho_f*((1./k)*inner(u-u1,phi)*dx \
+ inner(grad(u0)*(u - w), phi) * dx
+ inner(sigma_fluid(p,u),grad(phi))*dx) - inner(div(u),eta)*dx

# Structure Variational form
#U = U1 + w*k

#G = rho_s*((1./k)*inner(w-w0,psi))*dx + inner(sigma_structure(U),grad(psi))*dx# \
#+ inner(u,v)*dx(2) - inner(w,v)*dx(2)

# Mesh movement, solving the equation laplace -nabla(grad(d))

#H = inner(grad(U),grad(psi))*dx(1) - inner(grad(U("-"))*n("-"),psi("-"))*dS(6)


a = lhs(F) #- lhs(G) - lhs(H)
L = rhs(F) #- rhs(G) - rhs(H)

#F1 = F-G-H

T= 1
t = 0.0
uwp = Function(VVQ)

u_file = File(".velocity/velocity.pvd")
u_file << u0
while t < T:
    #if MPI.rank(mpi_comm_world()) == 0:
    b = assemble(L)
    eps = 10
    k_iter = 0
    max_iter = 10
    while eps > 1E-6 and k_iter < max_iter:
        A = assemble(a)
        A.ident_zeros()
        [bc.apply(A,b) for bc in bcs]
        solve(A,uwp.vector(),b)
        u_,w_,p_ = uwp.split(True)
        eps = errornorm(u_,u0,degree_rise=3)
        k_iter += 1
        print "k: ",k_iter, "error: %.3e" %eps
        u0.assign(u_)
    #w0.assign(w_)
    #w_.vector()[:] *= float(k)
    #U1.vector = w_.vector()[:]
    #ALE.move(mesh,w_)
    #mesh.bounding_box_tree().build(mesh)

    plot(u_,interactive=True)

    u1.assign(u_)
    u_file << u_
    print "Time:",t

    t += dt

plot(u_,interactive=True)
