from dolfin import *

mesh = UnitSquareMesh(10, 10)
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)

VQ = MixedFunctionSpace([V, Q])

d_ = TestFunction(VQ)
d, _ = split(d_)
v,_ = TrialFunctions(VQ)
#from IPython import embed; embed()

F_Ext = inner(grad(d), grad(v))*dx

bc = DirichletBC(VQ.sub(0), Constant((1,1)), "on_boundary")

df1 = Function(VQ)
df, _ = df1.split(True)

#df =Function(V)
a = lhs(F_Ext)
L = rhs(F_Ext)

solve(a==L, df1, bc)
df_hold ,_ = df1.split(True)
#df.assign(df_hold)
plot(df,interactive = True)
