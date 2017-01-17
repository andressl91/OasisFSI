
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
domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bar_area.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
#plot(domains,interactive = True)
dx_f = dx(1,subdomain_data=domains)
dx_s = dx(2,subdomain_data=domains)

#FluidSpace
V = VectorFunctionSpace(mesh, "CG", v_deg)
Q = FunctionSpace(mesh, "CG", p_deg)
VQ = V*Q

psiphi = TestFunction(VQ)
psi, phi = split(psiphi)
up = TrialFunction(VQ);
u, p = split(up)
up0 = Function(VQ)
u0, p0 = split(up0)
up_sol = Function(VQ)

v = TestFunction(V)
u_hat = TrialFunction(V)
u_hat_sol = Function(V)
u0 = Function(V)
u0_hat = Function(V)

k = Constant(dt)
#f = Constant((0, 0, 0))
nu = Constant(mu/rho)


# Define boundary conditions
bcu = []
bcp = []

#StructureSpace
D = VectorFunctionSpace(mesh, "CG", d_deg)
U = VectorFunctionSpace(mesh, "CG", d_deg)
W = D*U
psi = TestFunction(W)
w_sol = Function(W)

wd = Function(W)
w, d = split(wd)
wd0 = Function(W)
w0, d0 = split(wd0)
wd_1 = Function(W)
w_1, d_1 = split(wd_1)

gamma = TestFunction(D)
d_eta = TrialFunction(D)
d_eta_sol = Function(D)
#d0 = Function(W); d_1 = Function(W)
w_meshvel = TrialFunction(D)
w_sol = Function(D)

d_tomove = Function(D)

# Get fluid variational formula
mu = 1; rho = 1; dt = 1.
k = Constant(dt)
nu = Constant(mu/rho)

# Step 0: Extrapolation of the fluid-structure interface
F_d = inner(d_eta - d0 + k*(3./2*w0 - 1./2*w_1), gamma)*dx

# Step 1: Definition of the new domain
F_meshvel = inner(w_meshvel - 1./k*(d_eta_sol - d0, gamma))*dx_s
F_smooth = inner(grad(w_meshvel), grad(gamma))*dx_f

#Step 2
# Advection-diffusion step (explicit coupling)
F1 = (1./k)*inner(u_hat - u0, psi)*dx + inner(grad(u_hat)*(u0_hat - w_sol), v)*dx + \
     2.*nu*inner(eps(u_hat), eps(v))*dx
a1 = lhs(F1); L1 = rhs(F1)

#Step 3
# Projection step(implicit coupling)
F2 = (rho/k)*inner(u - u_hat_sol, psi)*dx - inner(p, div(psi))*dx \
+ inner(div(u), phi)*dx \
+ inner(u -  )

a2 = lhs(F2); L2 = rhs(F2)


while dt < T:
    #Step 0:
    solve(lhs(F_d) == rhs(F_d), d_eta_sol, bcs_d)

    #Step 1:
    F = F_meshvel + F_smooth
    solve(lhs(F) == rhs(F), w_sol, bcs_w)
    d_tomove.vector()[:] += w_sol.vector()[:]*float(k)
    ALE.move(d_tomove)

    #Step 2 ALE-advection-diffusion step
    #fluid_solve(A1, A2, L1, L2, fluid_solver, pressure_solver)
    solve(a1 == L1, u1, bcs)
    solve(a2 == L2, )


count += 1
u0.assign(u1)
p0.assign(p1)
t += dt
