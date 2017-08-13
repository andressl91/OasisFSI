
from dolfin import *
import numpy as np
set_log_active(False)

I = Identity(2)
def F_(U):
	return (Identity(2) + grad(U))

def J_(U):
	return det(F_(U))
def E(U):
	return 0.5*(F_(U).T*F_(U)-I)

def S(U,lamda_s,mu_s):
	return (2*mu_s*E(U) + lamda_s*tr(E(U))*I)

def P1(U,lamda_s,mu_s):
	return F_(U)*S(U,lamda_s,mu_s)

N = 4
dt = 0.01
mesh = UnitSquareMesh(N, N)

x = SpatialCoordinate(mesh)

V = VectorFunctionSpace(mesh, "CG", 1)
W = V*V
n = FacetNormal(mesh)

du = Function(W)
d, u = split(du)

phi, psi = TestFunctions(W)
du0 = Function(W)
d0, u0 = split(du0)

k = Constant(dt)
t_step = dt

mu_s = 1
rho_s = 1
lamda_s = 1

d_x = x[0]; d_y = x[1]
u_x = 0; u_y = 0

dx = "x[0]"; dy = "x[0]"
ux = "0"; uy = "0"
d_e = Expression((dx, dy), degree = 2, domain=mesh)
u_e = Expression((ux, uy), degree = 2, domain=mesh)

assign(du0.sub(0), project(d_e, V))
assign(du0.sub(1), project(u_e, V))
u_vec = as_vector([u_x, u_y])
d_vec = as_vector([d_x, d_y])
# Create right hand side f
f1 =rho_s*diff(u_vec, t) - div(P1(d_vec,lamda_s,mu_s))
#f2 = diff(d_vec, t) - u_vec # is zero when d and u is created to be zero

delta = 1E10
F_lin = (rho_s/dt)*inner(u-u0,phi)*dx
F_lin = inner(u, phi)*dx
F_lin = delta*((1.0/k)*inner(d-d0,psi)*dx - inner(u,psi)*dx)
F_lin -= inner(f1, phi)*dx #+ inner(f2, psi)*dx
