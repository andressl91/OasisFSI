from dolfin import *

#FUNKER!!! :D #SOLVE part of mixedspace in own newtonsolver and then
#map dofs back to function in mixedspace
"""
mesh = UnitSquareMesh(20, 20)

W = VectorFunctionSpace(mesh, 'CG', 2)
P = FunctionSpace(mesh, 'CG', 1)

WP = MixedFunctionSpace([W, P])

up = Function(WP)
# Make a deep copy to create two new Functions u and p (not subfunctions of WP)
u, p = up.split(deepcopy=True)

f = Expression(('sin(x[0])', 'cos(x[1])'))

v = TestFunction(W)
F = dot(u, v) * dx - dot(f, v) * dx

solve(F == 0, u)

# Now use black magic to put newly computed vector back in mixed space.
# WP.sub(0).dofmap().collapse(mesh) returns the collapsed dofmap and a
# dictionary (->[1]) that maps the collapsed dofmap to the dofs in the mixed
# space, i.e. the final values()
d = WP.sub(0).dofmap().collapse(mesh)[1].values()
up.vector()[d] = u.vector()
"""
## FUNKER !!! :D
#Split function for mixedspace to functions, solve with linear solver and but back
#to mixedspace
"""
mesh = UnitSquareMesh(20, 20)

W = VectorFunctionSpace(mesh, 'CG', 2)
U = VectorFunctionSpace(mesh, "CG", 2)
D = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, 'CG', 1)

WUDP =  MixedFunctionSpace([W, U, D, P])

wudp = Function(WUDP)

# Make a deep copy to create two new Functions u and p (not subfunctions of WP)
w, u, d, p = wudp.split(deepcopy=True)

f = Expression(('sin(x[0])', 'cos(x[1])'))

u_t = TrialFunction(W)
v_t = TestFunction(W)
u_sol = Function(W)

v = TestFunction(W)
F = dot(u_t, v_t) * dx - dot(f, v) * dx

solve(lhs(F) == rhs(F), u_sol)

# Now use black magic to put newly computed vector back in mixed space.
# WP.sub(0).dofmap().collapse(mesh) returns the collapsed dofmap and a
# dictionary (->[1]) that maps the collapsed dofmap to the dofs in the mixed
# space, i.e. the final values()
d = WUDP.sub(1).dofmap().collapse(mesh)[1].values()
wudp.vector()[d] = u_sol.vector()
"""

# WORKS!!! :D
"""
mesh = UnitSquareMesh(20, 20)
V = VectorFunctionSpace(mesh, "CG", 1)
U = VectorFunctionSpace(mesh, "CG", 1)

W = MixedFunctionSpace([V, U])

u = TrialFunction(V)
v = TestFunction(V)

uv0 = Function(W)
u0, v0 = uv0.split(deepcopy=True)

uv1 = Function(W)
u1, v1 = uv1.split(deepcopy=True)

k = Constant(0.1)
F = inner(u - u0 - k*u1, v)*dx

bc1 = DirichletBC(V, Constant((0, 0)), "on_boundary")
bc2 = DirichletBC(V, Constant((0, 1)), "x[0] < DOLFIN_EPS")
bcs = [bc1, bc2]

u_sol = Function(V)

solve(lhs(F) == rhs(F), u_sol, bcs)
"""

mesh = UnitSquareMesh(20, 20)
V = VectorFunctionSpace(mesh, "CG", 1)
U = VectorFunctionSpace(mesh, "CG", 1)

W = MixedFunctionSpace([V, U])

up = TrialFunction(W)
u, p = split(up)
vq = TestFunction(W)
v, q = split(vq)

w = Function(V)

k = Constant(0.1)
F = inner(u - k*w, v)*dx

bc1 = DirichletBC(V, Constant((0, 0)), "on_boundary")
bc2 = DirichletBC(V, Constant((0, 1)), "x[0] < DOLFIN_EPS")
bcs = [bc1, bc2]

u_sol = Function(W)

solve(lhs(F) == rhs(F), u_sol, bcs)
