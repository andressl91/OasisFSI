from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

from parser import *
from mappings import *
from Hron_Turek import *
from variational_form import *
from Traction import *


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
df_file = XDMFFile(mpi_comm_world(), "new_results/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/df.xdmf")
p_file = XDMFFile(mpi_comm_world(), "new_results/FSI-" +str(FSI) +"/P-"+str(v_deg) +"/dt-"+str(dt)+"/pressure.xdmf")

for tmp_t in [u_file, d_file, p_file]:
    tmp_t.parameters["flush_output"] = True
    tmp_t.parameters["multi_file"] = 1
    tmp_t.parameters["rewrite_function_mesh"] = False


# TEST TRIAL FUNCTIONS
phi, gamma = TestFunctions(VQ)
psi = TestFunction(V1)
u, p  = TrialFunctions(VQ)

up_ = Function(VQ)
u_, p_  = up_.split()

uf = Function(V1)
pf = Function(Q)

up0 = Function(VQ)
u0, p0 = up0.split()
up_res = Function(VQ)

#XI = TestFunction(V1)
df = Function(V1)
df0 = Function(V1)

#df, _ = df_res.split(True)
d = TrialFunction(V1)
d__ = Function(VQ)
d_, _ = d__.split(True)
d0 = Function(V1)
d1 = Function(V1)
d2 = Function(V1)
traction_ = Function(V1)
d_res = Function(V1)

k = Constant(dt)
I = Identity(2)
delta = 1.0E8
alpha = 1.0
beta = 0.01
#h =  mesh.hmin()

time_list = []
dis_x = []
dis_y = []
Drag = []
Lift = []

class DF_BC(Expression):
    def eval(self, value, x):
        value[0] = d_(x)[0]
        value[1] = d_(x)[1]

    def value_shape(self):
        return (2,)

df_bc = DF_BC()

#Boundary conditions for df in Ext
noslip = Constant((0.0,0.0))
df_bar    = DirichletBC(V1, df_bc, boundaries, 5)
df_inlet  = DirichletBC(V1, noslip, boundaries, 3)
df_outlet = DirichletBC(V1, noslip, boundaries, 4)
df_wall   = DirichletBC(V1, noslip, boundaries, 2)
df_circ   = DirichletBC(V1, noslip, boundaries, 6) #No slip on geometry in fluid
df_barwall = DirichletBC(V1, noslip, boundaries, 7) #No slip on geometry in fluid
bc_df = [df_bar, df_wall, df_inlet, df_circ, df_barwall,  df_outlet]#


# Making boundary values of force as a function to be used in variational form.


#before_first_compute(u)
########

def sigma_f_new(u,p,d,mu_f):
	return -p*I + mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

#The operator B_h is very simple to implement.
#You simply need to add to the fluid matrix the diagonal entries of the solid lumped-mass matrix related to the solid nodes lying on the fluid-solid interface.
#Note that, when you add these entries to the fluid matrix, you have to take into account the mapping between fluid and solid interface nodes.
#Fluid domain update:
#Ext operator:

# Structure var form
af, bf, a_s, b_s, adf, Ldf = var_form(d,d0,d1,d2,d_,df,df0,phi,VQ,V1,u,u_,u0,traction_,gamma,psi,p,p_,beta,delta,Q,rho_s,rho_f,k,dx_s,dx_f,mu_f)

counter = 0
t = dt

time_script_list = []
# Newton parameters
atol = 1e-6;rtol = 1e-6; max_it = 100; lmbda = 1.0;

ones_d = Function(V1)
ones_u = Function(VQ)
ones_d.vector()[:] = 1.
ones_u.vector()[:] = 1.

mesh,n=get_mesh_n()
print "Dofs: ",VQ.dim(), "Cells:", mesh.num_cells()


print "n------->",n

while t <= T:
    print "Time t = %.5f" % t
    time_list.append(t)
    if t < 2:
        inlet.set_t(t)
    if t >= 2:
        inlet.set_t(2)


    #Adf = assemble(adf, keep_diagonal=True)
    #Adf.ident_zeros()
    #Bdf = assemble(Ldf)
    #print "b_s", type(b_s)

    #[bc.apply(Adf, Bdf) for bc in bc_df]
    #[bc.apply(Adf, Bdf, df.vector()) for bc in bc_df]

    #pc = PETScPreconditioner("default")
    #sol = PETScKrylovSolver("default",pc)

    #plot(df, interactive=True)
    #solve(Adf, df.vector(), Bdf)
    #plot(df)

    Mass_s_b = assemble((rho_s/k)*inner(df("-"), phi("-"))*dS(5))
    #Mass_u_b = assemble(delta*inner(u("-"), phi("-"))*dS(5), keep_diagonal=True)
    #Mass_s_b_rhs = assemble(delta*((1.0/k)*inner(d0("-")-d1("-"),phi("-"))*dS(5)), keep_diagonal=True)

    #Mass_s_b_rhs = assemble((rho_s/k)*inner((2*((d0("-")-d1("-"))/k) - ((d1("-") - d2("-"))/k)), phi("-"))*dS(5))
    #Mass_s_b_rhs = assemble(delta*((1.0/k)*inner(d0("-")-d1("-"),phi("-"))*dS(5)))


    Mass_s_b_L = Mass_s_b*ones_u.vector() #Mass structure matrix lumped

    #Mass_s_and_rhs = Mass_s_b_L*Mass_s_b_rhs
    #Mass_u_b_L = Mass_s_b_l*Mass_u_b

    mass_form = inner(u,phi)*dx
    M_lumped = assemble(mass_form)
    M_lumped.zero()
    M_lumped.set_diagonal(Mass_s_b_L)

    #df_hold, _ = df_res.split(True)
    #df.assign(df_res)
    #plot(df)#,interactive=True)
    # Solve fluid step, find u and p
    A = assemble(af, keep_diagonal=True)# + delta*Mass_u_b#, tensor=A) #+ Mass_s_b_L
    A += M_lumped

    A.ident_zeros()

    B = assemble(bf) #+ delta*Mass_s_b_rhs
    #B += Mass_s_b_rhs

    #B -= Mass_s_and_rhs

    #[bc.apply(A,b) for bc in bcs]
    #plot(u_, interactive=True)
    [bc.apply(A, B) for bc in (bc_u+bc_p)]
    #plot(u_, interactive=True)
    #pc = PETScPreconditioner("default")
    #sol = PETScKrylovSolver("default",pc)
    solve(A, up_.vector(), B)

    #up0.assign(up_)
    u_, p_ = up_.split(True)
    u0.assign(u_); p0.assign(p_)
    #plot(p_)
    #plot(d_)

    #Dr = -assemble((sigma_f_new(u_,p_,d_,mu_f)*n)[0]*ds(6))
    #Li = -assemble((sigma_f_new(u_,p_,d_,mu_f)*n)[1]*ds(6))
    #Dr += -assemble((sigma_f_new(u_('-'),p_('-'),d_('-'),mu_f)*n('-'))[0]*dS(5))
    #Li += -assemble((sigma_f_new(u_('-'),p_('-'),d_('-'),mu_f)*n('-'))[1]*dS(5))

    #print "Drag: %.2f Lift: %.2f" %(Dr,Li)
    #print Dr, Li




    #uf.assign(u_)
    #pf.assign(p_)
    #plot(u_, interactive=True)
    #u__.assign(u_)
    #p__.assign(p_)
    """
    Mass_s_rhs = assemble((rho_s/(k*k))*inner(-2*d0+d1, psi)*dx_s) #solid mass time
    Mass_s_lhs = assemble((rho_s/(k*k))*inner(d, psi)*dx_s)
    Mass_s_rhs_L = Mass_s_rhs*ones_d.vector() #Mass_time structure matrix lumped lhs
    Mass_s_lhs_L = Mass_s_lhs*ones_d.vector() #Mass_time structure matrix lumped rhs

    mass_time_form = inner(d,psi)*dx
    M_time_lumped_lhs = assemble(mass_time_form)
    M_time_lumped_lhs.zero()
    M_time_lumped_lhs.set_diagonal(Mass_s_lhs_L)
    #print type(M_time_lumped_lhs)
    M_time_lumped_rhs = assemble(mass_time_form)
    #print type(M_time_lumped_rhs)
    M_time_lumped_rhs.zero()
    M_time_lumped_rhs.set_diagonal(Mass_s_rhs_L)


    # Solve structure step find d
    #traction_.assign(boundary_compute(u_, d_, p_)) #updating traction.

    A_s = assemble(a_s,keep_diagonal=True) + M_time_lumped_lhs#, tensor=A) #+ Mass_s_b_L
    B_s = assemble(b_s) - Mass_s_rhs_L
    A_s.ident_zeros()
    #print "b_s", type(b_s)

    [bc.apply(A_s, B_s) for bc in bc_d]

    #pc = PETScPreconditioner("default")
    #sol = PETScKrylovSolver("default",pc)
    solve(A_s, d_.vector(), B_s)
    #solve(F_structure==0,d,bc_d)
    #d = Newton_manual_s(F_structure , d, bc_d, atol, rtol, max_it, lmbda,d_res)
    #df.assign(d_)
    #plot(d_,mode="displacement")#, interactive=True)
    """
    if counter%step == 0:
        plot(u_)
        u_file << u_
        d_file << d_
        df_file << df
        p_file << p_

    d2.assign(d1)
    d1.assign(d0)
    d0.assign(d_)
    df0.assign(df)
    t += dt
    counter +=1
print "mean time: ",np.mean(time_script_list)
"""
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
#plt.show()"""
