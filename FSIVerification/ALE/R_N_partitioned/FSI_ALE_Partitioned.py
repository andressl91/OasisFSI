from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

import argparse
from argparse import RawTextHelpFormatter
#from Problems import *
from parser import *
from mappings import *
from Hron_Turek import *



#time0 = time()
#parameters["num_threads"] = 2
parameters["allow_extrapolation"] = True
if args.refiner == None:
    print "None"
else:
    for i in range(args.refiner):
        mesh = refine(mesh)

#print "Dofs: ",VQ.dim(), "Cells:", mesh.num_cells()



# TEST TRIAL FUNCTIONS
pg = TestFunction(VQ)
df_ = Function(VQ)
df, _ = df_.split("True")
phi, gamma = split(pg)
psi = TestFunction(V2)
#u,d,w,p
u, p  = TrialFunctions(VQ)
up_ = Function(VQ)
u_, p_  = up_.split("True")
up0 = Function(VQ)
u0, p0 = up0.split("True")
up_res = Function(VQ)

#d = TrialFunction(V2)
d = TrialFunction(V2)
d_ = Function(V2)
d0 = Function(V2)
d1 = Function(V2)
d2 = Function(V2)
d_res = Function(V2)

k = Constant(dt)
print "Re = %f" % (Um/(mu_f/rho_f))
I = Identity(2)
delta = 1.0E10
alpha = 1.0
beta = 0.01
h =  mesh.hmin()

#The operator B_h is very simple to implement.
#You simply need to add to the fluid matrix the diagonal entries of the solid lumped-mass matrix related to the solid nodes lying on the fluid-solid interface.
#Note that, when you add these entries to the fluid matrix, you have to take into account the mapping between fluid and solid interface nodes.
#Fluid domain update:
#Ext operator:
F_Ext =  inner(grad(df), grad(psi))*dx_f #- inner(grad(d)*n, psi)*ds

# Structure var form
Mass_s_rhs = assemble((rho_s/(k*k))*inner(-2*d0+d1, psi)*dx_s) #solid mass time
Mass_s_lhs = assemble((rho_s/(k*k))*inner(d, psi)*dx_s)

Mass_s_b = assemble(inner(df("-"), phi("-"))*dS(5))
#from IPython import embed
#embed()

#Mass_s_b_lhs = assemble((rho_s/k)*inner(u("-"), phi("-"))*dS(5))
# TODO: Change d0 in V2 to VQ?
Mass_s_b_rhs = assemble((rho_s/k)*inner((2*((d0("-")-d1("-"))/k) - ((d1("-") - d2("-"))/k)), phi("-"))*dS(5))

ones_d = Function(V2)
ones_u = Function(VQ)
ones_d.vector()[:] = 1.
ones_u.vector()[:] = 1.
Mass_s_rhs_L = Mass_s_rhs*ones_d.vector() #Mass_time structure matrix lumped
Mass_s_lhs_L = Mass_s_lhs*ones_d.vector() #Mass_time structure matrix lumped
Mass_s_b_L = Mass_s_b*ones_u.vector() #Mass structure matrix lumped
#Mass_s_and_lhs = Mass_s_b_L*Mass_s_b_lhs
Mass_s_and_rhs = Mass_s_b_L*Mass_s_b_rhs

# Structure variational form
sigma = sigma_dev

F_structure = inner(sigma(d), epsilon(psi))*dx_s #+ ??alpha*(rho_s/k)*(0.5*(d-d1))*dx_s??
F_structure += delta*((1.0/k)*inner(d-d0,psi)*dx_s - inner(u_, psi)*dx_s)
F_structure += inner(sigma(d("-"))*n("-"), psi("-"))*dS(5) + inner(J_(d("-"))*2*mu_f*epsilon(u_("-"))*inv(F_(d("-")).T)*n("-"), psi("-"))*dS(5) \
            - inner(J_(d("-"))*p("-")*I*n("-"), psi("-"))*dS(5)

# Fluid variational form
F_fluid = (rho_f/k)*inner(J_(df)*(u - u0), phi)*dx_f
F_fluid += rho_f*inner(J_(df)*grad(u)*inv(F_(df))*(u0 - ((df-d0)/k)), phi)*dx_f
F_fluid += inner(J_(df)*2*mu_f*epsilon(u)*inv(F_(df)), J_(df)*epsilon(phi)*inv(F_(df)))*dx_f
F_fluid -= inner(J_(df)*p*I*inv(F_(df)).T, grad(phi))*dx_f
F_fluid -= inner(div(J_(df)*inv(F_(df)).T*u), gamma)*dx_f
F_fluid += inner(J_(df("-"))*2*mu_f*epsilon(u_("-"))*inv(F_(df("-")))*n("-"), epsilon(phi("-"))*F_(df("-")))*dS(5)
F_fluid -= inner(J_(df("-"))*p("-")*I*n("-"), phi("-"))*dS(5)
F_fluid += inner(sigma(df("-"))*n("-"), J_(df("-"))*epsilon(phi("-"))*F_(df("-")))*dS(5)

F_fluid += - beta*h*h*inner(J_(df)*grad(p)*inv(F_(df)), J_(df)*grad(gamma))*inv(F_(df))*dx_f

a = lhs(F_fluid)
b = rhs(F_fluid)
a_s = lhs(F_structure) #+ Mass_s_L#+ Mass_s_and_lhs
b_s = rhs(F_structure) #+ Mass_s_and_rhs


t = 0.0
time_list = []

u_file = XDMFFile(mpi_comm_world(), "new_results/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/velocity.xdmf")
d_file = XDMFFile(mpi_comm_world(), "new_results/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/d.xdmf")
p_file = XDMFFile(mpi_comm_world(), "new_results/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/pressure.xdmf")

for tmp_t in [u_file, d_file, p_file]:
    tmp_t.parameters["flush_output"] = True
    tmp_t.parameters["multi_file"] = 1
    tmp_t.parameters["rewrite_function_mesh"] = False

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
    time_list.append(t)
    if t < 2:
        inlet.t = t;
    if t >= 2:
        inlet.t = 2;

    #Update fluid domain, solving laplace d = 0, solve for d_star?????????
    solve(F_Ext == 0 , d, bc_d)

    # Solve fluid step, find u and p
    A = assemble(a) #,keep_diagonal=True)#, tensor=A) #+ Mass_s_b_L
    A += Mass_s_b_L
    A.ident_zeros()
    b = assemble(b)
    b += Mass_s_b_L + Mass_s_and_rhs

    #[bc.apply(A,b) for bc in bcs]
    [bc.apply(A, b, up_.vector()) for bc in bc_u]

    #pc = PETScPreconditioner("default")
    #sol = PETScKrylovSolver("default",pc)
    solve(A, up_.vector(), b)

    up0.assign(up_)
    u_, p_ = up_.split(True)

    # Solve structure step find d
    A_s = assemble(a_s, keep_diagonal=True) + Mass_s_L#, tensor=A) #+ Mass_s_b_L
    b_s = assemble(b_s)
    #[bc.apply(A,b) for bc in bcs]
    [bc.apply(A_s, b_s, d_.vector()) for bc in bc_d]

    #pc = PETScPreconditioner("default")
    #sol = PETScKrylovSolver("default",pc)
    solve(A_s, d_.vector(), b_s)
    #solve(F_structure==0,d,bc_d)
    #d = Newton_manual_s(F_structure , d, bc_d, atol, rtol, max_it, lmbda,d_res)

    if counter%step==0:
        #if MPI.rank(mpi_comm_world()) == 0:
        #u_file << u
        #d_file << d
        #p_file << p
        #print "u-norm:",norm(u),"d-norm:", norm(d),"p-norm:",norm(p)
        """Dr = -assemble((sigma_f_hat(u,p,d)*n)[0]*ds(6))
        Li = -assemble((sigma_f_hat(u,p,d)*n)[1]*ds(6))

        #print 't=%.4f Drag/Lift on circle: %g %g' %(t,Dr,Li)
        #print 'INNER: t=%.4f Drag/Lift on circle: %g %g' %(t,Dr,Li)

        Dr += -assemble((sigma_f_hat(u('-'),p('-'),d('-'))*n('-'))[0]*dS(5))
        Li += -assemble((sigma_f_hat(u('-'),p('-'),d('-'))*n('-'))[1]*dS(5))
        #print 't=%.4f Drag/Lift : %g %g' %(t,Dr,Li)
        Drag.append(Dr)
        Lift.append(Li)

        #print "Drag: %.4f , Lift: %.4f  "%(integrateFluidStress(p, u))
        #print "x_bar: ", d(coord)[0], "y_bar: ",d(coord)[1]
        dsx = d(coord)[0]
        dsy = d(coord)[1]
        dis_x.append(dsx)
        dis_y.append(dsy)

        if MPI.rank(mpi_comm_world()) == 0:
            print "t = %.4f " %(t)
            print 'Drag/Lift : %g %g' %(Dr,Li)
            print "dis_x/dis_y : %g %g "%(dsx,dsy)"""
    d2.assign(d1)
    d1.assign(d0)
    d0.assign(d_)
    plot(d_)
    t += dt
    counter +=1
print "mean time: ",np.mean(time_script_list)
#print "script time: ", time.time()-time0
plt.plot(time_list,dis_x); plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
plt.savefig("results/FSI-" +str(FSI_deg) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/dis_x.png")
#plt.show()
plt.plot(time_list,dis_y);plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
plt.savefig("results/FSI-" +str(FSI_deg) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/dis_y.png")
#plt.show()
plt.plot(time_list,Drag);plt.ylabel("Drag");plt.xlabel("Time");plt.grid();
plt.savefig("results/FSI-" +str(FSI_deg) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/drag.png")
#plt.show()
plt.plot(time_list,Lift);plt.ylabel("Lift");plt.xlabel("Time");plt.grid();
plt.savefig("results/FSI-" +str(FSI_deg) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/lift.png")
#plt.show()
