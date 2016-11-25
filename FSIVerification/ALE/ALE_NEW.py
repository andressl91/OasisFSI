from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
parameters["allow_extrapolation"] = True
mesh = Mesh("fluid_new.xml")


for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break


V1 = VectorFunctionSpace(mesh, "CG", 2) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 2) # displacement
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VVQ = MixedFunctionSpace([V1, V2, Q])

# BOUNDARIES

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))
Bar = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.21)) or near(x[1], 0.19) or near(x[0], 0.6 ) )
Circle =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  ))
Barwall =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

Allboundaries = DomainBoundary()

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"


boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
Barwall.mark(boundaries, 7)
#Bar_area.mark(boundaries, 8)
#plot(boundaries,interactive=True)


ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh)


domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bar_area.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
#plot(domains,interactive = True)
dx_f = dx(1,subdomain_data=domains)
dx_s = dx(2,subdomain_data=domains)

#BOUNDARY CONDITIONS
# FLUID
FSI = 3
nu = 10**-3
rho_f = 1.0*1e3
mu_f = rho_f*nu
U_in = [0.2, 1.0, 2.0][FSI-1]   # Reynolds vel

# SOLID
Pr = 0.4
mu_s = [0.5, 0.5, 2.0][FSI-1]*1e6
rho_s = [1.0, 10, 1.0][FSI-1]*1e3
lamda_s = 2*mu_s*Pr/(1-2.*Pr)
Um = U_in
H = 0.41
L = 2.5
# "
#inlet = Expression(("(1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2))*(1-cos(t*pi/2.0))/2.0" \
#,"0"), t = 0.0, Um = Um, H = H)
class inlet(Expression):
	def __init__(self):
		self.t = 0
	def eval(self,value,x):
		value[0] = 0.5*(1-np.cos(self.t*np.pi/2))*1.5*Um*x[1]*(H-x[1])/((H/2.0)**2)
		value[1] = 0
	def value_shape(self):
		return (2,)
inlet = inlet()
#Fluid velocity conditions
u_inlet  = DirichletBC(VVQ.sub(0), inlet, boundaries, 3)
u_wall   = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 2)
u_circ   = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
u_bar    = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 5) #No slip on geometry in fluid
u_barwall= DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

#displacement conditions:
d_wall    = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 2)
d_inlet   = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 3)
d_outlet  = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 4)
d_circle  = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 6)
d_barwall = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid


#deformation condition
#d_barwall = DirichletBC(VVQ.sub(2), ((0, 0)), boundaries, 7)

#Pressure Conditions
p_out = DirichletBC(VVQ.sub(2), 0, boundaries, 4)
#p_bar = DirichletBC(VVQ.sub(2), Constant(0), boundaries, 8)

#Assemble boundary conditions
bcs = [u_inlet, u_wall, u_circ, u_barwall,\
       d_wall, d_inlet, d_outlet, d_circle,d_barwall,\
       p_out]#,p_bar]
# AREAS




# TEST TRIAL FUNCTIONS
phi, psi, gamma = TestFunctions(VVQ)
#u,d,w,p
#u,d, p  = TrialFunctions(VVQ)

udp = Function(VVQ)
udp0 = Function(VVQ)

udp_res = Function(VVQ)

u, d, p  = split(udp)
#u0, d0, p0  = split(udp0)

#d = Function(V)
d0 = Function(V2)
u0 = Function(V1)

dt = 0.0003
k = Constant(dt)
#EkPa = '62500'
#E = Constant(float(EkPa))



"""
#Fluid properties
rho_f   = Constant(1.0E3)
nu_f = Constant(1.0E-3)
mu_f    = Constant(1.0)

#Structure properties
rho_s = 1.0E3
mu_s = 0.5E6
nu_s = 0.4
E_1 = 1.4E6
lamda_s = nu_s*2*mu_s/(1-2*nu_s)
g = Constant((0,-2*rho_s))"""

print "Re = %f" % (Um/(mu_f/rho_f))

def integrateFluidStress(p, u):
  eps   = 0.5*(grad(u) + grad(u).T)
  sig   = -p*Identity(2) + 2.0*mu_f*eps

  traction  = dot(sig, -n)

  forceX = traction[0]*ds(5) + traction[0]*ds(6)
  forceY = traction[1]*ds(5) + traction[1]*ds(6)
  fX = assemble(forceX)
  fY = assemble(forceY)

  return fX, fY
def Newton_manual(F, udp, bcs, atol, rtol, max_it, lmbda,udp_res):
    #Reset counters
    Iter      = 0
    residual   = 1
    rel_res    = residual
    dw = TrialFunction(VVQ)
    Jac = derivative(F, udp,dw)                # Jacobi

    while rel_res > rtol and residual > atol and Iter < max_it:
        A = assemble(Jac)
        A.ident_zeros()
        b = assemble(-F)

        [bc.apply(A, b, udp.vector()) for bc in bcs]

        #solve(A, udp_res.vector(), b, "superlu_dist")

        solve(A, udp_res.vector(), b)#, "lu")

        udp.vector()[:] = udp.vector()[:] + lmbda*udp_res.vector()[:]
        #udp.vector().axpy(1., udp_res.vector())
        [bc.apply(udp.vector()) for bc in bcs]
        rel_res = norm(udp_res, 'l2')
        residual = b.norm('l2')

        if MPI.rank(mpi_comm_world()) == 0:
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
        % (Iter, residual, atol, rel_res, rtol)
        Iter += 1

    return udp

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

"""def s_s_n_l(U):
    #I = Identity(2)
    F_ = I + grad(U)
    E = 0.5*((F_.T*F_)-I)
    return lamda_s*tr(E)*I + 2*mu_s*E
def sigma_fluid(p, u): #NEWTONIAN FLUID
    #I = Identity(2)
    #F_ = I + grad(u)
    return -p*Identity(2) + mu_f *(inv(F_)*grad(u)+grad(u).T*inv(F_.T))"""


delta = 1.0E10
h =  mesh.hmin()
#d = d0 + k*u
#I = Identity(2)
#F_ = I + grad(d0)
#J = det(F_)
# Fluid variational form
F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx_f \
        + rho_f*inner(J_(d)*inv(F_(d))*grad(u)*(u - ((d-d0)/k)), phi)*dx_f \
        + inner(sigma_f_hat(u,p,d), grad(phi))*dx_f \
        - inner(div(J_(d)*inv(F_(d).T)*u), gamma)*dx_f\
        #- 0.5*h*h*inner(J_(d)*inv(F_(d))*grad(p),grad(gamma))*dx_f

        #- 0.05*h**2*inner(grad(p),grad(gamma))*dx_f
       #- inner(J*sigma_fluid(p,u)*inv(F_.T)*n, phi)*ds

# Structure var form
F_structure = (rho_s/k)*inner(J_(d)*(u-u0),phi)*dx_s + inner(P1(d),grad(phi))*dx_s

# Setting w = u on the structure using (d-d0)/k = w
F_w = delta*((1.0/k)*inner(d-d0,psi)*dx_s - inner(u,psi)*dx_s)

# laplace
F_laplace =  (1./k)*inner(d-d0,psi)*dx_f +inner(grad(d), grad(psi))*dx_f #- inner(grad(d)*n, psi)*ds

F = F_fluid + F_structure + F_w + F_laplace

T = 3.0
t = 0.0
time = []

u_file = File("mvelocity/abacus/P2-P2_3/velocity.pvd")
d_file = File("mvelocity/abacus/P2-P2_3/d.pvd")
p_file = File("mvelocity/abacus/P2-P2_3/pressure.pvd")

#[bc.apply(udp0.vector()) for bc in bcs]
#[bc.apply(udp.vector()) for bc in bcs]


dis_x = []
dis_y = []
counter = 0
t = dt
while t <= T:
    time.append(t)
    print "Time t = %.5f" % t

    if t < 2:
        inlet.t = t;
    if t >= 2:
        inlet.t = 2;

    #J1 = J(d0)
    #Reset counters
    atol = 1e-7;rtol = 1e-7; max_it = 100; lmbda = 1.0;
    udp = Newton_manual(F, udp, bcs, atol, rtol, max_it, lmbda,udp_res)

    #solve(lhs(F)==rhs(F),udp,bcs)
    u,d,p = udp.split(True)

    #udp0.assign(udp)

    #plot(u)
    if counter%10==0:
        if MPI.rank(mpi_comm_world()) == 0:
            u_file <<u
            d_file <<d
            print "u-norm:",norm(u),"d-norm:", norm(d),"p-norm:",norm(p)
            Dr = -assemble((sigma_f_hat(u,p,d)*n)[0]*ds(6))
            Li = -assemble((sigma_f_hat(u,p,d)*n)[1]*ds(6))
            print 't=%.4f Drag/Lift on circle: %g %g' %(t,Dr,Li)

            Dr += -assemble((sigma_f_hat(u('-'),p('-'),d('-'))*n('-'))[0]*dS(5))
            Li += -assemble((sigma_f_hat(u('-'),p('-'),d('-'))*n('-'))[1]*dS(5))
            print 't=%.4f Drag/Lift : %g %g' %(t,Dr,Li)

            #print "Drag: %.4f , Lift: %.4f  "%(integrateFluidStress(p, u))

            dis_x.append(d(coord)[0])
            dis_y.append(d(coord)[1])
    u0.assign(u)
    d0.assign(d)
    t += dt
    counter +=1
plt.plot(time,dis_x,); plt.ylabel("Displacement x-P2-P2_3");plt.xlabel("Time");plt.grid();
plt.savefig("results/abacus/dis_x.png")
plt.plot(time,dis_y,);plt.ylabel("Displacement y-P2-P2_3");plt.xlabel("Time");plt.grid();
plt.savefig("results//abacus/dis_y.png")
plt.show()
