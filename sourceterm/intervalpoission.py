from ufl import *
from dolfin import *
import matplotlib.pyplot as plt

mesh = IntervalMesh(100, 0, 1)
x = SpatialCoordinate(mesh)
cell = interval
element = FiniteElement('Lagrange', mesh, 2)
W = FunctionSpaceBase(mesh, element)

u = TrialFunction(W)
v = TestFunction(W)
u_sol = Function(W)

u_e = Expression("1/6.*x[0]*x[0]*x[0]")

u_x = 1./6*x[0]*x[0]*x[0]#Works!! Constant(2)
# Annotate expression w as a variable that can be used by "diff"
#u_x = variable(u_x)
#When using SpatialCoordinate, it seems variable must not be defined
F_x = div(grad(u_x))
F = F_x

a = -inner(grad(u), grad(v))*dx
L = inner(F, v)*dx
u_sol = Function(W)
bcs = DirichletBC(W, u_e, "on_boundary")
solve(a == L, u_sol, bcs)
print "ERRORNORM", errornorm(u_e, u_sol, norm_type="l2", degree_rise = 2)
u_e = interpolate(u_e, W)
#plot(u_sol)
#plot(u_e)
#interactive()

mesh_points=mesh.coordinates()
x0=mesh_points[:,0]
y = u_sol.vector().array()[:]
exact = u_e.vector().array()[:]
print len(x0), len(y), len(exact)
plt.figure(1)
plt.plot(x0, y, label="Numerical")
plt.plot(x0, exact, label="Exact")
plt.legend()
plt.show()
