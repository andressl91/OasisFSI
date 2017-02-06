from ufl import *
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

def MMS(N):
    mesh = UnitSquareMesh(N, N, "right")
    x = SpatialCoordinate(mesh)
    cell = interval
    #element = FiniteElement("Lagrange", cell, 1)
    element = VectorElement('Lagrange', mesh, 1)
    W = FunctionSpaceBase(mesh, element)

    u = TrialFunction(W)
    v = TestFunction(W)
    u_sol = Function(W)

    u_e = Expression(("x[0]*x[0]", "x[1]*x[1]"))
    u_x = x[0]*x[0]
    u_y = x[1]*x[1] #Works!! Constant(2)
    u_vec = as_vector([u_x, u_y])
    #pl = interpolate(start, W)
    #plot(start, interactive=True)

    # Annotate expression w as a variable that can be used by "diff"
    #u_x = variable(u_x)
    #When using SpatialCoordinate, it seems variable must not be defined
    F = div(grad(u_vec))


    a = -inner(grad(u), grad(v))*dx
    L = inner(F, v)*dx
    u_sol = Function(W)
    bcs = DirichletBC(W, u_e, "on_boundary")
    solve(a == L, u_sol, bcs)
    #u_e = interpolate(u_e, W)

    error = errornorm(u_e, u_sol, norm_type="l2", degree_rise = 3)
    print "ERRORNORM", error
    return error , N

N_list = [4,8,16,32,64,128,256]
E = np.zeros(len(N_list))
h = np.zeros(len(N_list))
for j in range(len(N_list)):
    E[j],h[j] = MMS(N_list[j])
    #print "Error: %2.2E ,dt: %.f: "%(E[j],dt_list[j])
for i in range(1, len(E)):
    r = np.log(E[i]/E[i-1])/np.log(h[i]/h[i-1])
    print "h= %10.2E , r= %.6f" % (h[i], r)
for k in range(len(E)):
    print "Error: ", E[k]



#plot(u_sol)
#plot(u_e)
#interactive()
"""
mesh_points=mesh.coordinates()
x0 = mesh_points[:,0]
y = u_sol.vector().array()[:]
exact = u_e.vector().array()[:]
print len(x0), len(y), len(exact)
plt.figure(1)
plt.plot(x0, y, label="Numerical")
plt.plot(x0, exact, label="Exact")
plt.legend()
plt.show()
"""
