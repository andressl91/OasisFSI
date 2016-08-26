from dolfin import *
import numpy as np
set_log_active(False)

mesh = Mesh("von_karman_street.xml")
V = VectorFunctionSpace(mesh,"CG", 2)
Q = FunctionSpace(mesh,"CG", 1)
print "Dofs: ", V.dim() + Q.dim()

u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

class Circle(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and not (near(x[0],0) or near(x[0],2.2) or near(x[1],0) or near(x[1],0.41))

class Right(SubDomain):
	def inside(self,x,on_boundary):
		return near(x[0], 2.2)

class Left(SubDomain):
	def inside(self,x,on_boundary):
		return near(x[0], 0)
class Nos(SubDomain):
	def inside(self,x,on_boundary):
		return near(x[1], 0) or near(x[1],0.41)


circle = Circle()
left = Left()
right = Right()
nos = Nos()
bound = FacetFunction("size_t", mesh)
bound.set_all(0)
nos.mark(bound, 3)
circle.mark(bound, 1)
left.mark(bound,4)
right.mark(bound,2)
plot(bound); interactive()
Um = 0.3
H = 0.41
inlet = Expression(("4*Um*x[1]*(H-x[1])/(H*H)", "0"),t=0.0,Um = Um,H=H)
U_mean = 2.0*Um/3.0
v_theta = Expression(("0","0"))

bc1 = DirichletBC(V, inlet , left)
bc2 = DirichletBC(V, (0, 0), nos)
bc3 = DirichletBC(V, v_theta , circle)
bc4 = DirichletBC(Q, 0.0,right)
bcs = [bc1,bc2,bc3]
bcp = [bc4]

u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)

dt = 0.01
nu = 0.001
rho_f = 1000.0
mu_f = rho_f*nu

k = Constant(dt)
f = Constant((0, 0))

# first without Pressure
F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx +  nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# correction with Pressure
a2 = -k*inner(grad(p),grad(q))*dx
L2 = div(u1)* q *dx

# last step

a3 = inner(u,v)*dx
L3 = inner(u1,v)*dx - k*inner(grad(p1),v)*dx


ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

def integrateFluidStress(u, p):

    eps   = 0.5*(nabla_grad(u) + nabla_grad(u).T)
    sig   = -p*Identity(len(u)) + 2.0*mu_f*eps

    traction  = dot(sig, n)

    forceX  = traction[0]*ds(1)
    forceY  = traction[1]*ds(1)
    fX      = assemble(forceX)*2/(U_mean**2*0.1)
    fY      = assemble(forceY)*2/(U_mean**2*0.1)

    return fX, fY



T = 1.0
t = dt
counter = 0
Drag = []
Lift = []
del_p = []
time = []
x0 = np.where(mesh.coordinates()[:,0]==0.15)
x1 = np.where(mesh.coordinates()[:,0]==0.25)
while t < T + DOLFIN_EPS:
	# Update pressure boundary condition
	inlet.t = t
	solve(a1==L1,u1,bcs)

	#pressure correction
	solve(a2==L2,p1,bcp)
	#print norm(p1)

	#last step
	solve(a3==L3,u1,bcs)

	u0.assign(u1)
	print "Timestep: ", t



	"""if (counter%100)==0:
		ufile << u1
		pfile << p1
		plot(u1,rescale = True)
		plot(p1,rescale = True)
		print "Counter: ",counter"""
	R = VectorFunctionSpace(mesh, 'R', 0)
	c = TestFunction(R)
	tau = -p1*Identity(2)+mu_f*(grad(u1)+grad(u1).T)
	n = FacetNormal(mesh)
	ds = ds[bound]
	forces = -assemble(dot(dot(tau, n), c)*ds(1)).array()*2/(U_mean**2*0.1)
	Drag.append(forces[0]); Lift.append(forces[1])
	print "Drag: ",forces[0],"Lift: ", forces[1]
	print "------------------------"
	drag,lift =integrateFluidStress(u1, p1)
	print "Time: ",t ," drag: ",drag, "lift: ",lift
	print "------------------------"


	p_ = p1.compute_vertex_values()
	diff_p = p_[x0[0]]-p_[x1[0]]
	del_p.append(diff_p)
	time.append(t)

	counter+=1
	t += dt
Drag[0] = 0; Lift[0] = 0
print "max Drag",max(Drag), "max Lift",max(Lift)
np.savetxt('drag.txt', Drag, delimiter=',')
np.savetxt('left.txt', Lift, delimiter=',')
np.savetxt('pressure.txt', del_p, delimiter=',')
np.savetxt('time.txt', time, delimiter=',')
