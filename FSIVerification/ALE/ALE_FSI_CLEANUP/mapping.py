from dolfin import *

I = Identity(2)

def Eij(U):
	return sym(grad(U))# - 0.5*dot(grad(U),grad(U))

def F_(U):
	return (I + grad(U))

def J_(U):
	return det(F_(U))

def E(U):
	return 0.5*(F_(U).T*F_(U)-I)

def S(U,lamda_s,mu_s):
	return (2*mu_s*E(U) + lamda_s*tr(E(U))*I)

def P1(U,lamda_s,mu_s):
	return F_(U)*S(U,lamda_s,mu_s)

def sigma_f(v,p,mu_f):
	return 2*mu_f*sym(grad(v)) - p*Identity(2)

def sigma_f_hat(v,p,u,mu_f):
	return J_(u)*sigma_f(v,p,mu_f)*inv(F_(u)).T
def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)
def sigma_f_new(u,p,d,mu_f):
	return -p*I + mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)
