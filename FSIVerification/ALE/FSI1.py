from dolfin import *
from parse import *
import matplotlib.pyplot as plt
time0 = time()
args = parse()
v_deg = args.v_deg
p_deg = args.p_deg
d_deg = args.d_deg
T = args.T
dt = args.dt
beta = args.beta
step = args.step
fig = False

from mapping import *
from calculations import *
from NewtonSolver import *
from Hron_Turek_problem import *
from solver_monolithic import *

mesh = Mesh("fluid_new.xml")
if args.refiner == None:
    print "None"
else:
    for i in range(args.refiner):
        mesh = refine(mesh)
V1 = VectorFunctionSpace(mesh, "CG", v_deg) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", d_deg) # displacement
Q  = FunctionSpace(mesh, "CG", p_deg)       # Fluid Pressure

VVQ = MixedFunctionSpace([V1, V2, Q])
print "Dofs: ",VVQ.dim(), "Cells:", mesh.num_cells()


#BOUNDARY CONDITIONS
# FLUID
nu = 1.0E-3
rho_f = 1.0E3
mu_f = rho_f*nu

# SOLID
Pr = 0.4
H = 0.41
L = 2.5

U_in = 0.2
mu_s = 0.5E6
rho_s = 1.0E3
lamda_s= 2*mu_s*Pr/(1-2.*Pr)

# Getting boundary conditions from problem file.
bcs,dx_f,dx_s,ds,dS,n,inlet,coord,boundaries = Hron_Turek_bcs(VVQ,mesh,U_in,H)

# Getting var-form from general monolithic solver.
F,udp, udp_res,d0 , d1 ,u0 , p0 = monolithic_form(VVQ,V1,V2,Q,dx_f,dx_s,mesh,v_deg,beta,n,lamda_s,mu_s,rho_f ,mu_f ,rho_s,dt)

t = 0.0
time_list = []

u_file = XDMFFile(mpi_comm_world(), "FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/velocity.xdmf")
d_file = XDMFFile(mpi_comm_world(), "FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/d.xdmf")
p_file = XDMFFile(mpi_comm_world(), "FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/pressure.xdmf")

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

timeme = []


"""
ref: 14.295 0.7638
     0.0227 0.8209
P2-P2-P1:
Drag/Lift : 14.1735 0.762091
dis_x/dis_y : 2.27409e-05 0.000799326
F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx_f
F_fluid += rho_f*inner(J_(d)*grad(u)*inv(F_(d))*(u - ((d-d0)/k)), phi)*dx_f
F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx_f
F_fluid += inner(J_(d)*sigma_f_new(u,p,d,mu_f)*inv(F_(d)).T, grad(phi))*dx_f


"""





while t <= T:
    print "Time t = %.5f" % t
    time_list.append(t)
    if t < 2:
        inlet.t = t;
    if t >= 2:
        inlet.t = 2;

    #Reset counters
    atol = 1e-6;rtol = 1e-6; max_it = 100; lmbda = 1.0;
    timeN = time()
    udp = Newton_manual(F, udp, bcs, atol, rtol, max_it, lmbda,udp_res,VVQ)
    timeme.append(time()-timeN)
    print "Newton time: %.2f" %(time()-timeN)

    u,d,p = udp.split(True)

    #plot(u)
    if counter%step==0:
        u_file << u
        d_file << d
        p_file << p

        Dr,Li = integrateFluidStress(u,p,d,mu_f,n,ds(6),dS(5))
        Drag.append(Dr)
        Lift.append(Li)

        dsx = d(coord)[0]
        dsy = d(coord)[1]
        dis_x.append(dsx)
        dis_y.append(dsy)

        if MPI.rank(mpi_comm_world()) == 0:
            print "t = %.4f " %(t)
            print 'Drag/Lift : %g %g' %(Dr,Li)

            print "dis_x/dis_y : %g %g "%(dsx,dsy)
    u0.assign(u)
    d1.assign(d0)
    d0.assign(d)
    p0.assign(p)
    t += dt
    counter +=1
print "average time: ", np.mean(timeme)
print "script time: ", time()-time0
plt.plot(time_list,dis_x); plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
plt.savefig("FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/dis_x.png")
#plt.show()
plt.plot(time_list,dis_y);plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
plt.savefig("FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/dis_y.png")
#plt.show()
plt.plot(time_list,Drag);plt.ylabel("Drag");plt.xlabel("Time");plt.grid();
plt.savefig("FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/drag.png")
#plt.show()
plt.plot(time_list,Lift);plt.ylabel("Lift");plt.xlabel("Time");plt.grid();
plt.savefig("FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/lift.png")
#plt.show()
