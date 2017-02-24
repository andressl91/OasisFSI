from fenics import *

mesh = UnitSquareMesh(10, 10)
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)

VQ = MixedFunctionSpace([V, Q])

d = TestFunction(VQ)
d, _ = split(d)
v = TrialFunction(VQ)
from IPython import embed; embed()

F_Ext = inner(grad(d), grad(v))*dx

bc = DirichletBC(V, Constant((0,0)), "on_boundary")

df = Function(V)

a = lhs(F_Ext)
L = rhs(F_Ext)

solve(a==L, df, bc)
