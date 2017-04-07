from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, mpi_comm_world
#from semi_implicit import *


def extrapolate_setup(F_fluid_linear, extype, mesh_file, d_, phi, gamma, dx_f, **semimp_namespace):
    def F_(U):
    	return Identity(len(U)) + grad(U)

    def J_(U):
    	return det(F_(U))
    if extype == "det":
        #alfa = inv(J_(d_["n"]))
        alfa = 1./(J_(d_["n"]))
    if extype == "smallconst":
        alfa = 0.01*(mesh_file.hmin())**2
    if extype == "const":
        alfa = 1.0

    F_extrapolate = alfa*inner(grad(d_["n"]), grad(phi))*dx_f
    F_fluid_linear += F_extrapolate

    return dict(F_fluid_linear=F_fluid_linear)
