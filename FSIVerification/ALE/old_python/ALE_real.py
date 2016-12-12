from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
mesh = Mesh("fluid_new.xml")


for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break


V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # Mesh movement
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VVQ = MixedFunctionSpace([V1, V2, Q])

# BOUNDARIES

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))
Bar = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.21)) or near(x[1], 0.19) or near(x[0], 0.6 ) )
Circle =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  ))
#Barwall =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

Allboundaries = DomainBoundary()

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
#Barwall.mark(boundaries, 7)
#plot(boundaries,interactive=True)


ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh)

#BOUNDARY CONDITIONS

Um = 2.0
H = 0.41
L = 2.5
# "
inlet = Expression(("(1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2))*(1-cos(t*pi/2.0))/2.0" \
,"0"), t = 0.0, Um = Um, H = H)

#Fluid velocity conditions
u_inlet  = DirichletBC(VVQ.sub(0), inlet, boundaries, 3)
u_wall   = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 2)
u_circ   = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
u_bar    = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 5) #No slip on geometry in fluid
u_barwall= DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid



#Mesh velocity conditions
w_wall    = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 2)
w_inlet   = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 3)
w_outlet  = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 4)
w_circle  = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 6)
#w_barwall = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 7)
#w_bar     = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 5)


#deformation condition
#d_barwall = DirichletBC(VVQ.sub(2), ((0, 0)), boundaries, 7)

#Pressure Conditions
#p_out = DirichletBC(VVQ.sub(2), 0, boundaries, 4)

#Assemble boundary conditions
bcs = [u_inlet, u_wall, u_circ,\
       w_wall, w_inlet, w_outlet, w_circle]#,w_bar]
# AREAS

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"

domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bar_area.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
#plot(domains,interactive = True)
dx_f = dx(1,subdomain_data=domains)
dx_s = dx(2,subdomain_data=domains)

# TEST TRIAL FUNCTIONS
phi, psi, gamma = TestFunctions(VVQ)

#u, w, p  = TrialFunctions(VVQ)

uwp = Function(VVQ)
u, w, p  = split(uwp)

u0 = Function(V1);
#d = Function(V2)
d0 =  Function(V2);

dt = 0.01
k = Constant(dt)
#EkPa = '62500'
#E = Constant(float(EkPa))

#Fluid properties
rho_f   = Constant(1.0E3)
nu_f = Constant(1.0E-3)
mu_f    = Constant(1.0)

#Structure properties
rho_s = 1.0E3
mu_s = 2.0E6
nu_s = 0.4
E_1 = 1.4E6
lamda_s = nu_s*2*mu_s/(1-2*nu_s)
g = Constant((0,-2*rho_s))

print "Re = %f" % (Um/(mu_f/rho_f))


def s_s_n_l(U):
    I = Identity(2)
    F = I + grad(U)
    E = 0.5*((F.T*F)-I)
    return F*(lamda_s*tr(E)*I + 2*mu_s*E)

def sigma_fluid(p, u): #NEWTONIAN FLUID
    return -p*Identity(2) + 2*mu_f * sym(grad(u))
def sigma_structure(d): #NEWTONIAN FLUID
    return 2*mu_s*sym(grad(d)) + lamda_s*tr(sym(grad(d)))*Identity(2)
def eps(u):
    return sym(grad(u))

delta = 1.0E-8
#d = d0 + k*u
I = Identity(2)
F = I + grad(d0)
J = det(F)
# Fluid variational form
F_fluid = J*(rho_f/k)*inner(u - u0,phi)*dx_f +  J*rho_f*inner(dot(grad(u),inv(F)*(u - w)), phi)*dx_f \
+ inner(J*sigma_fluid(p,u)*inv(F.T),grad(phi))*dx_f \
- inner(div(J*u*inv(F.T)),gamma)*dx_f

#Displacement velocity is equal to mesh velocity on dx_s
F_last = (1./delta)*inner(u,psi)*dx_s - (1./delta)*inner(w,psi)*dx_s

#Laplace
F_laplace = k*inner(grad(w),grad(psi))*dx_f+inner(grad(d0),grad(psi))*dx_f \
	       #- k*inner(grad(w('-'))*n('-'),psi('-'))*dS(5)\
	       #+ inner(grad(d0('-'))*n('-'),psi('-'))*dS(5)

# Structure Variational form
F_structure = J*(rho_s/k)*inner(u-u0,phi)*dx_s  \
 + inner(J*sigma_structure(d0)*inv(F.T),grad(phi))*dx_s + k*inner(J*sigma_structure(u)*inv(F.T),grad(phi))*dx_s



F = F_fluid + F_structure  + F_laplace + F_last
a = lhs(F)
L = rhs(F)


T = 20.0
t = 0.0
time = np.linspace(0,T,(T/dt)+1)

u_file = File("mvelocity/velocity.pvd")
w_file = File("mvelocity/w.pvd")
p_file = File("mvelocity/pressure.pvd")

#uwp1 = Function(VVQ)
#b = assemble(L)
solver = "Newton2"
dis_x = []
dis_y = []
counter = 0
while t <= T:
    if MPI.rank(mpi_comm_world()) == 0:
        print "Time t = %.3f" % t

    if t < 2:
        inlet.t = t;
    if t >= 2:
        inlet.t = 2;
    if solver == "Newton2":
        dw = TrialFunction(VVQ)
        dG_W = derivative(F, uwp, dw)                # Jacobi

        atol, rtol = 1e-7, 1e-10                    # abs/rel tolerances
        lmbda      = 1.0                            # relaxation parameter
        WD_inc     = Function(VVQ)                  # residual
        bcs_u      = []                             # residual is zero on boundary, (Implemented if DiriBC != 0)
        for i in bcs:
            i.homogenize()
            bcs_u.append(i)
        Iter      = 0                               # number of iterations
        residual  = 1                              # residual (To initiate)
        rel_res    = residual                       # relative residual
        max_it    = 100                             # max iterations
        #ALTERNATIVE TO USE IDENT_ZEROS()
        a = lhs(dG_W) + lhs(F);
        L = rhs(dG_W) + rhs(F);


        while rel_res > rtol and Iter < max_it:
            #ALTERNATIVE TO USE IDENT_ZEROS()
            A = assemble(a); b = assemble(L)
            A.ident_zeros()
            [bc.apply(A,b) for bc in bcs_u]
            solve(A,WD_inc.vector(),b)

            #WORKS!!
            #A, b = assemble_system(dG_W, -G, bcs_u)
            #solve(A, WD_inc.vector(), b)
            rel_res = norm(WD_inc, 'l2')

            #a = assemble(G)
            #for bc in bcs_u:
                #bc.apply(a)

            uwp.vector()[:] += lmbda*WD_inc.vector()

            Iter += 1
        print "Iterations: ",Iter," Relativ res: " , rel_res

        for bc in bcs:
            bc.apply(uwp.vector())


    u,w,p = uwp.split(True)
    w.vector()[:] *= float(k)
    d0.vector()[:] += w.vector()[:]
    plot(u,mode = "Displacement")
    u0.assign(u)
    if counter%10==0:
        u_file <<u
    dis_x.append(d0(coord)[0])
    dis_y.append(d0(coord)[1])

    t += dt
    counter +=1
plt.plot(time,dis_x,);title; plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
plt.plot(time,dis_y,);title; plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
