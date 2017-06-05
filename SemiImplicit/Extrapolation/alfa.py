from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, tr, norm, \
MPI, mpi_comm_world, CellVolume
#from semi_implicit import *


def extrapolate_setup(extype, mesh_file, dvp_, w, beta, gamma, dx_f, **semimp_namespace):
    def F_(U):
    	return Identity(len(U)) + grad(U)

    def J_(U):
    	return det(F_(U))
    def eps(U):
        return 0.5*(grad(U)*inv(F_(U)) + inv(F_(U)).T*grad(U).T)
    def STVK(U, alfa_mu, alfa_lam):
        return alfa_lam*tr(eps(U))*Identity(len(U)) + 2.0*alfa_mu*eps(U)
        #return F_(U)*(alfa_lam*tr(eps(U))*Identity(len(U)) + 2.0*alfa_mu*eps(U))

    alfa = 1.0 # holder value if linear is chosen
    if extype == "det":
        #alfa = inv(J_(d_["n"]))
        alfa = 1./(J_(d_["n"]))
    if extype == "smallconst":
        alfa = 0.01*(mesh_file.hmin())**2
    if extype == "const":
        alfa = 1.0

    F_extrapolate = alfa*inner(grad(w), grad(beta))*dx_f \
                    - inner(Constant((0, 0)), beta)*dx_f


    return dict(F_extrapolate=F_extrapolate)
