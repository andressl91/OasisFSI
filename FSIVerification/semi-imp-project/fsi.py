
from dolfin import *
from Utils.argpar import *
from Fluid_solver.projection import *
parameters['allow_extrapolation']=True

args = parse()
v_deg = args.v_deg
p_deg = args.p_deg
d_deg = args.d_deg
mesh = IntervalMesh(100, 0, 1)

#FluidSpace
V = VectorFunctionSpace(mesh, "CG", v_deg)
Q = FunctionSpace(mesh, "CG", p_deg)

u = TrialFunction(V); u_tilde = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

u0 = Function(V); u0_tilde = Function(V)
u1 = Function(V)
p0 = Function(Q)
p1 = Function(Q)

#FluidSpace
W = VectorFunctionSpace(mesh, "CG", v_deg)
w = TrailFunction(W)


# Define boundary conditions
bcu = []
bcp = []

#StructureSpace
W = VectorFunctionSpace(mesh, "CG", d_deg)

d = TrialFunction(W)
psi = TestFunction(W)

# Get fluid variational formula
mu = 1; rho = 1; dt = 1.
k = Constant(dt)
nu = Constant(mu/rho)

# Assemble matrices
a1, L1, a2, L2, a3, L3 = projection(k, nu, u, u_tilde, u0, u0_tildev, p, p1, q, w)
A1 = assemble(a1); A2 = assemble(a2); A3 = assemble(a3)

pc = PETScPreconditioner("jacobi")
fluid_solver = PETScKrylovSolver("bicgstab", pc)

pc2 = PETScPreconditioner("hypre_amg")
pressure_solver = PETScKrylovSolver("gmres", pc2)

while dt < T:

    fluid_solve(A1, A2, A3, L1, L2, L3, fluid_solver, pressure_solver)

count += 1
u0.assign(u1)
p0.assign(p1)
t += dt
