from fenics import *
from math import log
#from Newton import *

error = []
h = []
N_list = [3, 5, 10, 15]
for N in N_list:
    # Mesh and spatial coordinates
    mesh = UnitCubeMesh(N, N, N)
    x = SpatialCoordinate(mesh)

    # Function space
    W = VectorFunctionSpace(mesh, "CG", 1)

    # Define functions
    u = TrialFunction(W)
    v = TestFunction(W)
    u_sol = Function(W)

    # Define exact solution
    u_e = Expression((
                    "sin(pow(x[0], 4))",            # x-direction
                    "cos(pow(x[1], 4))",            # y-direction
                    "cos(pow(x[2], 4))*sin(x[2])"   # z-direction
                     ), degree=4)
    u_x = sin(x[0]**4)
    u_y = cos(x[1]**4)
    u_z = cos(x[2]**4)*sin(x[2])
    u_vec = as_vector([u_x, u_y, u_z])

    # Create right hand side f
    f = div(grad(u_vec))

    # Solve for f and exact bc
    a = -inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx
    u_sol = Function(W)
    bcs = DirichletBC(W, u_e, "on_boundary")
    solve(a == L, u_sol, bcs)

    error.append(errornorm(u_e, u_sol, norm_type="l2", degree_rise=3))
    h.append(mesh.hmin())

print "Convergence rate:"
for i in range(len(N_list) - 1):
    print log(error[i] / error[i+1]) / log(h[i] / h[i+1])
