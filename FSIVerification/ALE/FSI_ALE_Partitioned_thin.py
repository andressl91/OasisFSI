from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time
import sys


time0 = time.time()
#parameters["num_threads"] = 2
parameters["allow_extrapolation"] = True
mesh = UnitSquareMesh(10, 10)
#x = m.coordinates()

V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 2) # displacement
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VQ = MixedFunctionSpace([V1, Q])
#VVQ = MixedFunctionSpace([V1, V2, Q])
print "Dofs: ",VQ.dim(), "Cells:", mesh.num_cells()
# BOUNDARIES

Top = AutoSubDomain(lambda x: "on_boundary" and near(x[1],1))
Bottom = AutoSubDomain(lambda x: "on_boundary" and near(x[1],0))
NOS = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],0) or near(x[0],1)))

#Allboundaries = DomainBoundary()

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Top.mark(boundaries, 1)
Bottom.mark(boundaries, 2)
NOS.mark(boundaries, 3)
#plot(boundaries,interactive=True)

ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)

n = FacetNormal(mesh)
domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bottom.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
dx_f = dx(1,subdomain_data=domains)
dx_s = dx(2,subdomain_data=domains)


#BOUNDARY CONDITIONS
# FLUID
nu = 10**-3
rho_f = 1.0*1e3
mu_f = rho_f*nu
U_in = 1.0

# SOLID
Pr = 0.4
mu_s = 1e6
rho_s = 1e3
lamda_s = 2*mu_s*Pr/(1-2.*Pr)

#Fluid velocity conditions
u_top  = DirichletBC(VQ.sub(0), ((U_in,0.0)), boundaries, 1)
u_nos   = DirichletBC(VQ.sub(0), ((0.0, 0.0)), boundaries, 3)
#u_bottom   = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 2)

#displacement conditions:
d_top    = DirichletBC(V2, ((0.0, 0.0)), boundaries, 1)
d_nos   = DirichletBC(V2, ((0.0, 0.0)), boundaries, 3)
#d_outlet  = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 4)


#Assemble boundary conditions
bc_u = [u_top, u_nos]
bc_d = [d_nos]#d_top, d_nos]
bcs = bc_u + bc_d


# TEST TRIAL FUNCTIONS
phi, gamma = TestFunctions(VQ)
psi = TestFunction(V2)
#u,d,w,p
#u,d, p  = TrialFunctions(VVQ)

up = Function(VQ)
u, p  = split(up)
up0 = Function(VQ)
u0,p0 = split(up0)
up_res = Function(VQ)
d_res = Function(V2)


#d = TrialFunction(V2)
d = Function(V2)
d0 = Function(V2)
d1 = Function(V2)



dt = 0.01
k = Constant(dt)


def Newton_manual_s(F, d, bc_d, atol, rtol, max_it, lmbda,d_res):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    dw = TrialFunction(V2)
    Jac_1 = derivative(F, d,dw)                # Jacobi

    a = assemble(Jac_1)#,keep_diagonal)
    #a.vector().zero()
    while rel_res > rtol and residual > atol and Iter < max_it:
        A = assemble(Jac_1, tensor = a)
        #A.ident_zeros()
        b = assemble(-F)

        [bc.apply(A, b, d.vector()) for bc in bc_d]

        #solve(A, udp_res.vector(), b, "superlu_dist")

        solve(A, d_res.vector(), b)#, "mumps")

        d.vector()[:] = d.vector()[:] + lmbda*d_res.vector()[:]
        #udp.vector().axpy(1., udp_res.vector())
        [bc.apply(d.vector()) for bc in bc_d]
        rel_res = norm(d_res, 'l2')
        residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
        % (Iter, residual, atol, rel_res, rtol)
        Iter += 1

    return d

def Newton_manual(F, up, bcs, atol, rtol, max_it, lmbda,udp_res):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    dw = TrialFunction(VQ)
    Jac = derivative(F, up,dw)                # Jacobi

    a = assemble(Jac)
    #a.vector().zero()
    while rel_res > rtol and residual > atol and Iter < max_it:
        A = assemble(Jac, tensor = a)
        A.ident_zeros()
        b = assemble(-F)

        [bc.apply(A, b, up.vector()) for bc in bc_u]

        #solve(A, udp_res.vector(), b, "superlu_dist")

        solve(A, up_res.vector(), b)#, "mumps")

        up.vector()[:] = up.vector()[:] + lmbda*up_res.vector()[:]
        #udp.vector().axpy(1., udp_res.vector())
        [bc.apply(up.vector()) for bc in bc_u]
        rel_res = norm(up_res, 'l2')
        residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
        % (Iter, residual, atol, rel_res, rtol)
        Iter += 1

    return up

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


delta = 1.0E10
epsilon = 1.0E-6
alpha = 1.0
h =  mesh.hmin()

#Fluid domain update:
#Ext operator:
F_Ext =  inner(grad(d), grad(psi))*dx #- inner(grad(d)*n, psi)*ds

# Fluid variational form


F_fluid =(rho_f/k)*inner(J_(d)*(u - u0), phi)*dx \
        + rho_f*inner(J_(d)*inv(F_(d))*grad(u)*(u - ((d-d0)/k)), phi)*dx \
        + inner(sigma_f_hat(u,p,d), grad(phi))*dx \
        - inner(div(J_(d)*inv(F_(d).T)*u), gamma)*dx \
        + inner(J_(d)*sigma_f(u,p)*inv(F_(d)).T*n,phi)*ds(2) \
        + (rho_s/k)*epsilon*inner(u,phi)*ds(2)  \
        - (rho_s/k)*epsilon*inner(((d0-d1)/k) + k*((d-2*d0-d1)/(k*k)),phi)*ds(2) \
        - inner(J_(d0)*sigma_f(u0,p0)*inv(F_(d0)).T*n,phi)*ds(2)

# Structure var form

F_structure = (rho_s/k*k)*epsilon*inner(d-2*d0+d1,psi)*ds(2) \
            - inner(J_(d)*sigma_f(u,p)*inv(F_(d)).T*n,psi)*ds(2) \
            + delta*((1.0/k)*inner(d-d0,psi) - inner(u,psi))*ds(2)



T = 5.0
t = 0.0
time_list = []



u_file = File("results/FSI-THIN/velocity.pvd")
d_file = File("results/FSI-THIN/d.pvd")
p_file = File("results/FSI-THIN/pressure.pvd")

#[bc.apply(udp0.vector()) for bc in bcs]
#[bc.apply(udp.vector()) for bc in bcs]


dis_x = []
dis_y = []
Drag = []
Lift = []
counter = 0
t = dt

time_script_list = []

# Newton parameters
atol = 1e-6;rtol = 1e-6; max_it = 100; lmbda = 1.0;

while t <= T:
    print "Time t = %.5f" % t

    #Update fluid domain, solving laplace d = 0, solve for d_star?????????
    solve(F_Ext==0 , d, bc_d)

    # Solve fluid step, find u and p
    solve(F_fluid == 0, up,bc_u)
    up0.assign(up)
    u,p = up.split(True)
    #up = Newton_manual(F_fluid, up, bc_u, atol, rtol, max_it, lmbda,up_res)

    # Solve structure step find d
    solve(F_structure == 0 , d, bc_d)
    #udp = Newton_manual_s(F_structure, d, bc_d, atol, rtol, max_it, lmbda,d_res)
    #plot(u)#,interactive=True)
    u_file << u
    d_file << d
    p_file << p

    d1.assign(d0)
    d0.assign(d)
    plot(d)

    t += dt
    counter +=1
#print "mean time: ",np.mean(time_script_list)
