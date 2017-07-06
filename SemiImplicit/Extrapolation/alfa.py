from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, mpi_comm_world, CellVolume
#from semi_implicit import *


def extrapolate_setup(extype, mesh_file, d, w, phi, dx_f, **semimp_namespace):

    F_extrapolate = inner(grad(d), grad(phi))*dx_f \
                    - inner(Constant((0, 0)), phi)*dx_f

    return dict(F_extrapolate=F_extrapolate)
