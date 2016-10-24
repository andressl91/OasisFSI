from dolfin import *
parameters['allow_extrapolation']=True

#mesh = Mesh("von_karman_street_FSI_fluid.xml")
mesh = Mesh("fluid_new.xml")
#plot(mesh,interactive=True)

for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break

V1 = VectorFunctionSpace(mesh, "CG", 2) # Velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # Structure deformation
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VVQ = MixedFunctionSpace([V1,V2,Q])

# BOUNDARIES

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))
Bar = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.21)) or near(x[1], 0.19) or near(x[0], 0.6 ) )
Circle =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  ))
Barwall =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

Allboundaries = DomainBoundary()

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
#Allboundaries.mark(boundaries, 1)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
Barwall.mark(boundaries, 7)
#test = File("Facet.pvd")
#test << boundaries
#plot(boundaries,interactive=True)

testinterior = FacetFunction("size_t", mesh)
testinterior.set_all(0)
Bar.mark(testinterior, 5)


ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = testinterior)
#dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh)

#BOUNDARY CONDITIONS

################################################


Um = 1.0
H = 0.41
L = 2.5
D = 0.1

# t = 2.0 implies steady flow
inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2) * (1 - cos(t*pi/2))/2"\
,"0"), t = 0.0, Um = Um, H = H)

#velocity conditions
u_inlet  = DirichletBC(VVQ.sub(0), inlet, boundaries, 3)
u_wall   = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 2)
u_circ   = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 6) #No slip on geometry in fluid
u_barwall = DirichletBC(VVQ.sub(0), ((0, 0)), boundaries, 7)

#Deformation conditions
d_inlet  = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 3)
d_outlet  = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 4)
d_wall   = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 2)
d_circ   = DirichletBC(VVQ.sub(1), ((0, 0)), boundaries, 6) #No slip on geometry in fluid
d_barwall = DirichletBC(VVQ.sub(1),((0, 0)), boundaries, 7)

#Pressure Conditions
p_out = DirichletBC(VVQ.sub(2), 0, boundaries, 4)

#Assemble boundary conditions
bcs = [u_inlet, u_wall, u_circ, u_barwall, \
       d_inlet, d_wall, d_circ, d_barwall, d_outlet]
# AREAS

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"

domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bar_area.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
plot(domains,interactive = True)


# TEST TRIAL FUNCTIONS
psi, gamma, eta = TestFunctions(VVQ)
#u, w, p = TrialFunctions(VVQ)
udp = Function(VVQ)
u, d, p  = split(udp)
udp0 = Function(VVQ)
u0, d0, p0  = split(udp0)
d_disp = Function(V2)


#EkPa = '62500
#E = Constant(float(EkPa))

#Fluid properties
rho_f   = Constant(1000.0)
mu_f    = Constant(1.0)
nu_f = mu_f/rho_f

#Structure properties
#FSI 2
rho_s = 10.E3
mu_s = 0.5E6
nu_s = 0.4
E_1 = 1.4E6
lamda = nu_s*2*mu_s/(1-2*nu_s)
g = Constant((0,-2*rho_s))
dt = 0.001
k = Constant(dt)

Re = Um*D/nu_f
print "SOLVING FOR Re = %f" % Re #0.1 Cylinder diameter



def Venant_Kirchhof(d):
    I = Identity(2)
    F = I - grad(d)
    J = det(F)
    E = 0.5*((inv(F.T)*inv(F))-I)
    return inv(F)*(2.*mu_s*E + lamda*tr(E)*I)*inv(F.T)

def sigma_f(p, u):
  return - p*Identity(2) + mu_f*(grad(u) + grad(u).T)


# Fluid variational form
Fluid_momentum = (rho_f/k)*inner(u - u0, psi)*dx(1) + rho_f*inner(grad(u)*u, psi)*dx(1)+ \
    inner(sigma_f(p, u), grad(psi))*dx(1)

Fluid_continuity = eta*div(u)*dx(1)

#############################
I = Identity(2)
F_ = I - grad(d)
J_ = det(F_)
F_1 = I - grad(d0)
J_1 = det(F_1)
#theta = Constant(0.593)
theta = Constant(1)
# Structure Variational form
Solid_momentum = ( J_*rho_s/k*inner(u - u0, psi) \
    + rho_s*( J_*theta*inner(dot(grad(u), u), psi) + J_1*(1 - theta)*inner(dot(grad(u0), u0), psi) ) \
    + inner(J_*theta*Venant_Kirchhof(d) + (1 - theta)*J_1*Venant_Kirchhof(d0) , grad(psi))  \
    - (theta*J_*inner(g, psi) + (1-theta)*J_1*inner(g, psi) ) ) * dx(2)

Solid_deformation = dot(d - d0 + k*(theta*dot(grad(d), u) + (1-theta)*dot(grad(d0), u0) ) \
    - k*(theta*u + (1 -theta)*u0 ), gamma)  * dx(2)

# Mesh velocity function in fluid domain
d_smooth = inner(grad(d),grad(gamma))*dx(1) #- inner(grad(u("-"))*n("-"),phi("-"))*dS(5)

#Conservation of dynamics (CHECK SIGNS!!)
#dynamic = inner(Venant_Kirchhof(d('+'))*n('+'), psi('+'))*dS(5) + inner(sigma_f(p('-'), u('-'))*n('-') ,psi('-'))*dS(5)
#dynamic = - inner(Venant_Kirchhof(d('+'))*n('+'), psi('+'))*dS(5) - inner(sigma_f(p('-'), u('-'))*n('-') ,psi('-'))*dS(5)
dynamic = - inner(Venant_Kirchhof(d('-'))*n('-'), psi('-'))*dS(5) - inner(sigma_f(p('+'), u('+'))*n('+') ,psi('+'))*dS(5)
F = Fluid_momentum + Fluid_continuity \
  + Solid_momentum + Solid_deformation \
  + d_smooth + dynamic

t = 0
T = 10
time = []; dis_x = []; dis_y = []
vel_file = File("./velocity/velocity.pvd")
#vel_file << u0
while t <= T:
    print "Time %f" % t
    time.append(t)

    if t < 2:
        inlet.t = t;
    if t >= 2:
        inlet.t = 2;

    dw = TrialFunction(VVQ)
    dF_W = derivative(F, udp, dw)                # Jacobi

    atol, rtol = 1e-7, 1e-6                  # abs/rel tolerances
    lmbda      = 1.0                            # relaxation parameter
    WD_inc      = Function(VVQ)                  # residual
    bcs_u      = []                             # residual is zero on boundary, (Implemented if DiriBC != 0)
    for i in bcs:
        i.homogenize()
        bcs_u.append(i)
    Iter      = 0                               # number of iterations
    residual   = 1                              # residual (To initiate)
    rel_res    = residual                       # relative residual
    max_it    = 100                             # max iterations
    #ALTERNATIVE TO USE IDENT_ZEROS()
    a = lhs(dF_W) + lhs(F);
    L = rhs(dF_W) + rhs(F);
    ##NEWTON TALK! GOOD SOURCE
    #https://fenicsproject.org/qa/536/newton-method-programmed-manually
    while rel_res > rtol and Iter < max_it:
        #ALTERNATIVE TO USE IDENT_ZEROS()
        #A = assemble(a); b = assemble(L)
        A, b = assemble_system(dF_W, -F, bcs_u)
        A.ident_zeros()
        [bc.apply(A,b) for bc in bcs_u]
        solve(A,WD_inc.vector(),b)

        #WORKS!!
        #A, b = assemble_system(dG_W, -G, bcs_u)
        #solve(A, WD_inc.vector(), b)
        rel_res = norm(WD_inc, 'l2')

        #a = assemble(F)
        for bc in bcs_u:
            bc.apply(A)

        udp.vector()[:] += lmbda*WD_inc.vector()
        print "iteration %d, relative error %g" % (Iter, rel_res)
        Iter += 1


    for bc in bcs:
        bc.apply(udp.vector())

    u, d, p  = udp.split(True)

    u0, d0, p0  = udp0.split(True)

    d_disp.vector()[:] = d.vector()[:] - d0.vector()[:]
    ALE.move(mesh, d_disp)
    mesh.bounding_box_tree().build(mesh)
    udp0.assign(udp)

    vel_file << u

    #dis_x.append(d(coord)[0])
    #dis_y.append(d(coord)[1])

    t += dt



plt.figure(1)
plt.title("LIFT \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
plt.xlabel("Time Seconds")
plt.ylabel("Lift force Newton")
plt.plot(time, Lift, label='dt  %g' % dt)
plt.legend(loc=4)
plt.savefig("lift.png")

plt.figure(2)
plt.title("DRAG \n Re = %.1f, dofs = %d, cells = %d" % (Re, U_dof, mesh_cells))
plt.xlabel("Time Seconds")
plt.ylabel("Drag force Newton")
plt.plot(time, Drag, label='dt  %g' % dt)
plt.legend(loc=4)
plt.savefig("drag.png")
#plt.show()


#drag, lift = integrateFluidStress(p_, u_)
#print('Drag = %f, Lift = %f' \
	#%  (drag, lift))
