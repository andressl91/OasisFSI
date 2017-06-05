from dolfin import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(10, 10)
V = VectorFunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
u_sol = Function(V)

a = inner(grad(u), grad(v))*dx
L = inner(Constant((0,0)), v)*dx

# Create SubDomain and mark left hand side
Left = AutoSubDomain(lambda x, on_bnd: near(x[0], 0))
Right = AutoSubDomain(lambda x, on_bnd: near(x[0], 1))

facet = FacetFunction('size_t', mesh)
facet.set_all(0)
DomainBoundary().mark(facet, 1)
Left.mark(facet, 2)
Right.mark(facet, 3)

bc1 = DirichletBC(V, Constant((1, 1)), facet, 2)
bc2 = DirichletBC(V, Constant((1, 1)), facet, 3)
bc3 = DirichletBC(V, Constant((0, 0)), facet, 1)
bcs = [bc1, bc2, bc3]

solve(a == L, u_sol, bcs)
test = grad(u_sol)
J = det(test)
#print J
print assemble(J*dx)

plt.figure(1)
plt.plot(u_sol.vector().array())
plt.savefig("./res.png")
