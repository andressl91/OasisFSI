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
u_file = XDMFFile(mpi_comm_world(), "new_results/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/velocity.xdmf")
d_file = XDMFFile(mpi_comm_world(), "new_results/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/d.xdmf")
p_file = XDMFFile(mpi_comm_world(), "new_results/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/pressure.xdmf")

for tmp_t in [u_file, d_file, p_file]:
    tmp_t.parameters["flush_output"] = True
    tmp_t.parameters["multi_file"] = 1
    tmp_t.parameters["rewrite_function_mesh"] = False

#print "Dofs: ",VQ.dim(), "Cells:", mesh.num_cells()



# TEST TRIAL FUNCTIONS
df_res = Function(VQ)
df_1, _ = df_res.split()
df,_ = TrialFunctions(VQ)
phi, gamma = TestFunctions(VQ)
# = split(pg)
psi = TestFunction(V1)
#u,d,w,p
u, p  = TrialFunctions(VQ)
up_ = Function(VQ)
u_, p_  = up_.split()
#u__ = Function(V1)
#p__ = Function(Q)
up0 = Function(VQ)
u0, p0 = up0.split()
up_res = Function(VQ)

#d = TrialFunction(V2)
d = TrialFunction(V1)
d__ = Function(VQ)
d_, _ = d__.split()
d0 = Function(V1)
d1 = Function(V1)
d2 = Function(V1)
d_res = Function(V1)

k = Constant(dt)
print "Re = %f" % (Um/(mu_f/rho_f))
I = Identity(2)
delta = 1.0E10
alpha = 1.0
beta = 0.01
h =  mesh.hmin()
time_list = []
dis_x = []
dis_y = []
Drag = []
Lift = []


class DF_BC(Expression):
    def eval(self, value, x):
        value[0] = d_(x)[0]
        value[1] = d_(x)[1]
        #print "Value", value
        #print "d", d_(x)

    def value_shape(self):
        return (2,)

df_bc = DF_BC()

#The operator B_h is very simple to implement.
#You simply need to add to the fluid matrix the diagonal entries of the solid lumped-mass matrix related to the solid nodes lying on the fluid-solid interface.
#Note that, when you add these entries to the fluid matrix, you have to take into account the mapping between fluid and solid interface nodes.
#Fluid domain update:
#Ext operator:

# Structure var form
Mass_s_rhs = assemble((rho_s/(k*k))*inner(-2*d0+d1, psi)*dx_s) #solid mass time
Mass_s_lhs = assemble((rho_s/(k*k))*inner(d, psi)*dx_s)

Mass_s_b = assemble(inner(df("-"), phi("-"))*dS(5))

#Mass_s_b_lhs = assemble((rho_s/k)*inner(u("-"), phi("-"))*dS(5))
# TODO: Change d0 in V2 to VQ?
Mass_s_b_rhs = assemble((rho_s/k)*inner((2*((d0("-")-d1("-"))/k) - ((d1("-") - d2("-"))/k)), phi("-"))*dS(5))

ones_d = Function(V1)
ones_u = Function(VQ)
ones_d.vector()[:] = 1.
ones_u.vector()[:] = 1.
Mass_s_rhs_L = Mass_s_rhs*ones_d.vector() #Mass_time structure matrix lumped lhs
Mass_s_lhs_L = Mass_s_lhs*ones_d.vector() #Mass_time structure matrix lumped rhs

Mass_s_b_L = Mass_s_b*ones_u.vector() #Mass structure matrix lumped
#Mass_s_and_lhs = Mass_s_b_L*Mass_s_b_lhs
Mass_s_and_rhs = Mass_s_b_L*Mass_s_b_rhs


mass_form = inner(u,phi)*dx
M_lumped = assemble(mass_form)
M_lumped.zero()
M_lumped.set_diagonal(Mass_s_b_L)

mass_time_form = inner(d,psi)*dx
M_time_lumped_lhs = assemble(mass_time_form)
M_time_lumped_lhs.zero()
M_time_lumped_lhs.set_diagonal(Mass_s_lhs_L)
#print type(M_time_lumped_lhs)
M_time_lumped_rhs = assemble(mass_time_form)
#print type(M_time_lumped_rhs)
M_time_lumped_rhs.zero()
M_time_lumped_rhs.set_diagonal(Mass_s_rhs_L)


# Lifting operator
#df = TrialFunction(V1)
#phi = TestFunction(V1)
#d = Function(V1)
f_ = Function(VQ)
f, _ = f_.split()
F_Ext = inner(grad(df), grad(phi))*dx_f + inner(f, phi)*dx_f #- inner(grad(d)*n, psi)*ds
#df_ = Function(V1)
# Structure variational form
sigma = sigma_dev

F_structure = inner(sigma(d), grad(psi))*dx_s #+ ??alpha*(rho_s/k)*(0.5*(d-d1))*dx_s??
F_structure += delta*((1.0/k)*inner(d-d0,psi)*dx_s - inner(u_, psi)*dx_s)
#F_structure += inner(sigma(d("-"))*n("-"), psi("-"))*dS(5)
#F_structure += inner(J_(d("-"))*sigma_f_new(u_("-"),p_("-"),d("-"))*inv(F_(d("-"))).T*n("-"), psi("-"))*dS(5)

# Fluid variational form
F_fluid = (rho_f/k)*inner(J_(df)*(u - u0), phi)*dx_f
F_fluid += rho_f*inner(J_(df)*grad(u)*inv(F_(df))*(u0 - ((df-d0)/k)), phi)*dx_f
F_fluid += inner(J_(df)*sigma_f_new(u,p,df)*inv(F_(df)).T, grad(phi))*dx_f
F_fluid -= inner(div(J_(df)*inv(F_(df)).T*u), gamma)*dx_f
F_fluid += inner(J_(df("-"))*sigma_f_new(u("-"),p("-"),df("-"))*inv(F_(df("-"))).T*n("-"), phi("-"))*dS(5)
F_fluid += inner(sigma(df("-"))*n("-"), phi("-"))*dS(5)
F_fluid -= beta*h*h*inner(J_(df)*inv(F_(df).T)*grad(p), grad(gamma))*dx_f
#F_fluid -= beta*h*h*inner(J_(df)*grad(p)*inv(F_(df)), grad(gamma))*dx_f

#a = lhs(F_fluid)
#b = rhs(F_fluid)
#a_s = lhs(F_structure) #+ Mass_s_L#+ Mass_s_and_lhs
#b_s = rhs(F_structure) #+ Mass_s_and_rhs



#print "b_s", type(b_s)
counter = 0
t = dt
noslip = Constant((0.0,0.0))
df_bar    = DirichletBC(VQ.sub(0), df_bc, boundaries, 5)
df_inlet  = DirichletBC(VQ.sub(0), noslip, boundaries, 3)
df_wall   = DirichletBC(VQ.sub(0), noslip, boundaries, 2)
df_circ   = DirichletBC(VQ.sub(0), noslip, boundaries, 6) #No slip on geometry in fluid
df_barwall= DirichletBC(VQ.sub(0), noslip, boundaries, 7) #No slip on geometry in fluid
bc_df = [df_wall] #, df_inlet, df_circ, df_barwall, df_bar]#

time_script_list = []

a = lhs(F_Ext)
L = rhs(F_Ext)

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
    """print df_.vector().array().shape

    f = Function(V1)
    L = inner(f, phi)*dx_f
    A = assemble(a) #, keep_diagonal=True)
    B = assemble(L, keep_diagonal=True)"""
    solve(a == L, df_1, bc_df)

    # Solve fluid step, find u and p
    A = assemble(a) #,keep_diagonal=True)#, tensor=A) #+ Mass_s_b_L
    #A += Mass_s_b_L
    A += M_lumped
    A.ident_zeros()
    B = assemble(b)
    B += Mass_s_and_rhs

    #[bc.apply(A,b) for bc in bcs]
    [bc.apply(A, B, up_.vector()) for bc in bc_u]

    #pc = PETScPreconditioner("default")
    #sol = PETScKrylovSolver("default",pc)
    solve(A, up_.vector(), B)

    up0.assign(up_)
    u_, p_ = up_.split(True)
    #u__.assign(u_)
    #p__.assign(p_)

    # Solve structure step find d
    A_s = assemble(a_s, keep_diagonal=True) + M_time_lumped_lhs#, tensor=A) #+ Mass_s_b_L
    B_s = assemble(b_s) + Mass_s_rhs_L
    #print "b_s", type(b_s)

    #[bc.apply(A,b) for bc in bcs]
    [bc.apply(A_s, B_s, d_.vector()) for bc in bc_d]

    #pc = PETScPreconditioner("default")
    #sol = PETScKrylovSolver("default",pc)
    solve(A_s, d_.vector(), B_s)
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
    u_file << u_
    d_file << d_
    p_file << p_
    d2.assign(d1)
    d1.assign(d0)
    d0.assign(d_)
    plot(u_,interactive=True)
    print "WE REACHED THE END"
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
