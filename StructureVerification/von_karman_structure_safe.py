from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys
set_log_active(False)
"""if len(sys.argv) != 2:
    print "Usage: %s [implementation (1/2/3)]" % sys.argv[0]
    sys.exit(0)"""
implementation = sys.argv[1]
print implementation

mesh = Mesh("von_karman_street_FSI_structure.xml")

for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break

V = VectorFunctionSpace(mesh, "CG", 1)
if implementation == "2" or "4" or "5":
    VV=V*V
print "Dofs: ",V.dim(), "Cells:", mesh.num_cells()

BarLeftSide =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
BarLeftSide.mark(boundaries,1)
#plot(boundaries,interactive=True)

#PARAMETERS:
rho_s = 1.0E3
mu_s = 0.5E6
nu_s = 0.4
E_1 = 1.4E6
lamda = nu_s*2*mu_s/(1-2*nu_s)
g = Constant((0,-2*rho_s))
dt = float(sys.argv[2])
k = Constant(dt)
beta = Constant(0.25)

#Hookes
def sigma_structure(d):
    return 2*mu_s*sym(grad(d)) + lamda*tr(sym(grad(d)))*Identity(2)

#Cauchy Stress Tensor sigma
def Cauchy(d):
    I = Identity(2)
    F = I + grad(d)
    J = det(F)
    E = 0.5*((F.T*F)-I)
    return 1./J*F*(lamda*tr(E)*I + 2*mu_s*E )*F.T

def Venant_Kirchhof(d):
    I = Identity(2)
    F = I - grad(d)
    J = det(F)
    E = 0.5*((inv(F.T)*inv(F))-I)
    return inv(F)*(2.*mu_s*E + lamda*tr(E)*I)*inv(F.T)

#Second piola stress
def s_s_n_l_2(d):
    I = Identity(2)
    F = I + grad(d)
    E = 0.5*((F.T*F)-I)
    J = det(F)
    return (1/J)*F*(lamda*tr(E)*I + 2*mu_s*E)*F.T
def s_s_n_l(d):
    I = Identity(2)
    F = I + grad(d)
    E = 0.5*((F.T*F)-I)
    #J = det(F)
    return F*(lamda*tr(E)*I + 2*mu_s*E)


#Variational form with single space, just solving displacement
if implementation =="1":
    bc1 = DirichletBC(V, ((0,0)),boundaries, 1)
    bcs = [bc1]
    psi = TestFunction(V)
    d = Function(V)
    d0 = Function(V)
    d1 = Function(V)

    G =rho_s*((1./k**2)*inner(d - 2*d0 + d1,psi))*dx \
    + inner(0.5*(s_s_n_l(d)+s_s_n_l(d1)),grad(psi))*dx - inner(g,psi)*dx

#Variational form with double spaces
elif implementation =="2":
    bc1 = DirichletBC(VV.sub(0), ((0,0)),boundaries, 1)
    bc2 = DirichletBC(VV.sub(1), ((0,0)),boundaries, 1)
    bcs = [bc1,bc2]
    psi,phi = TestFunctions(VV)
    wd = Function(VV)
    w,d = split(wd)
    w0d0 = Function(VV)
    w0,d0 = split(w0d0)

    G =rho_s*((1./k)*inner(w-w0,psi))*dx + rho_s*inner(dot(grad(0.5*(w+w0)),0.5*(w+w0)),psi)*dx + inner(0.5*(s_s_n_l(d)+s_s_n_l(d0)),grad(psi))*dx \
    - inner(g,psi)*dx - dot(d-d0,phi)*dx + k*dot(0.5*(w+w0),phi)*dx

#Solving for displacement velocity and updating for displacement.
elif implementation == "3":
    bc1 = DirichletBC(V, ((0,0)),boundaries, 1)
    bcs = [bc1]
    psi = TestFunction(V)
    w = Function(V)
    w0 = Function(V)
    d0 = Function(V)
    d = Function(V)
    #d = d0 + w*k
    #d = d0+0.5*(w+w0)*k

    G =rho_s*((1./k)*inner(w-w0,psi))*dx \
    + inner(0.5*(s_s_n_l(d)+s_s_n_l(d0)),grad(psi))*dx - inner(g,psi)*dx

elif implementation =="4":
    bc1 = DirichletBC(VV.sub(0), ((0,0)),boundaries, 1)
    bc2 = DirichletBC(VV.sub(1), ((0,0)),boundaries, 1)
    bcs = [bc1,bc2]
    psi,phi = TestFunctions(VV)
    wd = Function(VV)
    w,d = split(wd)
    w0d0 = Function(VV)
    w0,d0 = split(w0d0)

    G = rho_s*inner(w-w0,psi)*dx + \
    (k/2.)*inner(s_s_n_l(d)+s_s_n_l(d0),grad(psi))*dx -k*inner(g,psi)*dx + \
        rho_s*inner(d-d0,phi)*dx - \
        rho_s*k*inner(w0,phi)*dx \
        + ((1-2*beta)*0.5)*k*k*(inner(s_s_n_l(d0),grad(phi))-inner(g,phi))*dx \
        + beta*k*k*(inner(s_s_n_l(d),grad(phi))-inner(g,phi))*dx


#Full Eulerian formulation with theta-rule scheme
elif implementation == "5":
    bc1 = DirichletBC(VV.sub(0), ((0,0)),boundaries, 1)
    bc2 = DirichletBC(VV.sub(1), ((0,0)),boundaries, 1)
    bcs = [bc1,bc2]
    psi, phi = TestFunctions(VV)
    wd = Function(VV)
    w, d = split(wd)
    w0d0 = Function(VV)
    w0, d0 = split(w0d0)
    d_disp = Function(V)

    I = Identity(2)
    F_ = I - grad(d) #d here must be the same as in variational formula
    J_ = det(F_)
    F_1 = I - grad(d0)
    J_1 = det(F_1)
    theta = Constant(0.593)
    #theta = Constant(0.61)

    G = ( J_*rho_s/k*inner(w - w0, psi) \
    + rho_s*( J_*theta*inner(dot(grad(w), w), psi) + J_1*(1 - theta)*inner(dot(grad(w0), w0), psi) ) \
    + inner(J_*theta*Venant_Kirchhof(d) + (1 - theta)*J_1*Venant_Kirchhof(d0) , grad(psi))  \
    - (theta*J_*inner(g, psi) + (1-theta)*J_1*inner(g, psi) ) ) * dx \
    + (dot(d - d0 + k*(theta*dot(grad(d), w) + (1-theta)*dot(grad(d0), w0) ) \
    - k*(theta*w + (1 -theta)*w0 ), phi) ) * dx

dis_x = []
dis_y = []

T = 10
t = 0
time = np.linspace(0,T,(T/dt)+1)
print "time len",len(time)
dis_file = File("results/x_direction.pvd")


from time import sleep
while t < T:
    print "Time: ",t
    if implementation == "1":
        #solve(G == 0, d, bcs, solver_parameters={"newton_solver": \
        #{"relative_tolerance": 1E-9,"absolute_tolerance":1E-9,"maximum_iterations":100,"relaxation_parameter":1.0}})
        solver="Newton2"
        if solver == "Newton2":
            dw = TrialFunction(V)
            dF_W = derivative(G, d, dw)                # Jacobi

            atol, rtol = 1e-12, 1e-12                    # abs/rel tolerances
            lmbda      = 1.0                            # relaxation parameter
            WD_inc     = Function(V)                  # residual
            bcs_u      = []                             # residual is zero on boundary, (Implemented if DiriBC != 0)
            for i in bcs:
                i.homogenize()
                bcs_u.append(i)
            Iter      = 0                               # number of iterations
            residual  = 1                              # residual (To initiate)
            rel_res    = residual                       # relative residual
            max_it    = 100                             # max iterations
            #ALTERNATIVE TO USE IDENT_ZEROS()
            #a = lhs(dG_W) + lhs(F);
            #L = rhs(dG_W) + rhs(F);


            while rel_res > rtol and Iter < max_it:
                #ALTERNATIVE TO USE IDENT_ZEROS()
                #A = assemble(a); b = assemble(L)
                A,b = assemble_system(dF_W,-G,bcs_u)
                A.ident_zeros()
                [bc.apply(A,b) for bc in bcs_u]
                solve(A,WD_inc.vector(),b)

                rel_res = norm(WD_inc, 'l2')

                #a = assemble(G)
                #for bc in bcs_u:
                #    bc.apply(A)

                d.vector()[:] += lmbda*WD_inc.vector()

                Iter += 1
            print "Iterations: ",Iter," Relativ res: " , rel_res

            for bc in bcs:
                bc.apply(d.vector())


        d1.assign(d0)
        d0.assign(d)
        plot(d,mode="displacement")
        dis_x.append(d(coord)[0])
        dis_y.append(d(coord)[1])

    elif implementation == "2":
        solve(G == 0, wd, bcs, solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
        w0d0.assign(wd)
        w0,d0 = w0d0.split(True)
        w,d = wd.split(True)
        #plot(d,mode="displacement")
        #w0.assign(w)
        #w.vector()[:] *= float(k)
        #d0.vector()[:] += w.vector()[:]
        #plot(d,mode="displacement")

        #ALE.move(mesh,w)
        #mesh.bounding_box_tree().build(mesh)
        #plot(mesh)
        dis_x.append(d(coord)[0])
        dis_y.append(d(coord)[1])


    elif implementation == "3":
        solve(G == 0, w, bcs, solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
        w0.assign(w)
        #d0.assign(d)
        w.vector()[:] *= float(k)
        d0.vector()[:] += w.vector()[:]
        #d0.assign(d)
        #ALE.move(mesh,w)
        #mesh.bounding_box_tree().build(mesh)
        plot(d0,mode="displacement")
        dis_x.append(d0(coord)[0])
        dis_y.append(d0(coord)[1])
        #w0.assign(w)

        #sleep(0.11)
    elif implementation == "4":
        solve(G == 0, wd, bcs, solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,"maximum_iterations":100,"relaxation_parameter":1.0}})
        w0d0.assign(wd)
        w0,d0 = w0d0.split(True)
        w,d = wd.split(True)
        dis_file << d
        #w0.assign(w)
        #w.vector()[:] *= float(k)
        #d0.vector()[:] += w.vector()[:]
        #plot(d,mode="displacement")

        #ALE.move(mesh,w)
        #mesh.bounding_box_tree().build(mesh)
        #plot(mesh)
        dis_x.append(d(coord)[0])
        dis_y.append(d(coord)[1])
    elif implementation == "5":
        solve(G == 0, wd, bcs, solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-9,"absolute_tolerance":1E-9,"maximum_iterations":100,"relaxation_parameter":1.0}})
        w,d = wd.split(True)
        w0,d0 = w0d0.split(True)
        d_disp.vector()[:] = d.vector()[:] - d0.vector()[:]
        ALE.move(mesh, d_disp)
        mesh.bounding_box_tree().build(mesh)
        w0d0.assign(wd)

        w0,d0 = w0d0.split(True)
        dis_x.append(d0(coord)[0])
        dis_y.append(d0(coord)[1])
    t += dt

print len(dis_x), len(time)
if implementation == "1":
    title = plt.title("Single space solving d")
elif implementation == "2":
    title = plt.title("Double space")
elif implementation == "3":
    title = plt.title("Single space solving w")
elif implementation == "4":
    title = plt.title("Newmark MixedFunctionSpace")
elif implementation == "5":
    title = plt.title("Eulerian MixedFunctionSpace")
plt.figure(1)
plt.title("Eulerian Mixed, schewed Crank-Nic")
plt.plot(time,dis_x,);title; plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
plt.savefig("run_x.jpg")
#plt.show()
plt.figure(2)
plt.title("Eulerian Mixed, schewed Crank-Nic")
plt.plot(time,dis_y);title;plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
plt.savefig("run_y.jpg")
#plt.show()
