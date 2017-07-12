from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, mpi_comm_world, CellVolume
#from semi_implicit import *


def Extrapolate_setup(d, phi, dx_f, d_, **semimp_namespace):
    alfa = 1. / det(Identity(len(d_["n"])) + grad(d_["n"]))
    F_extrapolate = alfa*inner(grad(d), grad(phi))*dx_f - inner(Constant((0, 0)), phi)*dx_f

    return dict(F_extrapolate=F_extrapolate)
