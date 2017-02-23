from dolfin import *
mu_s = 1
mu_f = 1
lamda_s = 1
I = Identity(2)
def Eij(U):
	return sym(grad(U))# - 0.5*dot(grad(U),grad(U))

def F_(U):
	return (I + grad(U))

def J_(U):
	return det(F_(U))

def E(U):
	return 0.5*(F_(U).T*F_(U)-I)

def S(U):
	return (2*mu_s*E(U) + lamda_s*tr(E(U))*I)

def P1(U):
	return F_(U)*S(U)

def sigma_f(v,p):
    	return 2*mu_f*sym(grad(v)) - p*Identity(2)

def sigma_s(u):
	return 2*mu_s*sym(grad(u)) + lamda_s*tr(sym(grad(u)))*I

def sigma_f_hat(v,p,u):
	return J_(u)*sigma_f(v,p)*inv(F_(u)).T

def sigma_dev(U): #linear solid stress tensor
	return 2*mu_s*sym(grad(U)) + lamda_s*tr(sym(grad(U)))*Identity(2)

def sigma_f_new(u,p,d):
	return -p*I + mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)
	
def epsilon(u):
    return sym(grad(u))
