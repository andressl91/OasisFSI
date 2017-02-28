from dolfin import *
from mapping import *

#Cauchy Stress Tensor Fluid
def sigma_f(p_, u_, mu):
    return -p_*Identity(2) + mu*(grad(u_) + grad(u_).T)

#Shear stress mapped for ALE
def sigma_f_shearstress_map(U, D):
    return grad(U)*inv(F_(D)) + inv(F_(D)).T*grad(U).T

#Test for sym(grad()) operator from UFL
def eps(u):
    return 0.5*(grad(u) + grad(u).T)
