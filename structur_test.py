from dolfin import *
set_log_level(PROGRESS)
mesh = Mesh("von_karman_street_FSI_structure.xml")
V = VectorFunctionSpace(mesh,"CG",1)
VV = V*V
#print mesh.coordinates().max()
for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001):
        print coord
        break

rho = Constant(1e3)
B = Constant((0,-2*rho)) #m/s^1

BarLeftSide =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
BarLeftSide.mark(boundaries,1)
plot(boundaries,interactive=True)
bc1 = DirichletBC(V, ((0,0)),boundaries, 1)
bcs = [bc1]
#plot(bound);interactive()
n = FacetNormal(mesh)
ds = Measure("ds", subdomain_data=boundaries)

up = Function(VV)
u, p = split(up)
v,q = TestFunctions(VV)


up0 = Function(VV)
u0,p0 = split(up0)
up1 = Function(VV)
u1,p1 = split(up1)

"""
u = Function(V)
du = TrialFunction(V)
v = TestFunction(V)"""
nu_s = 0.4
mu = Constant("0.5e6")
E = 1.4e6
#lam = Constant("0.0105e9")
lam = Constant(E*nu_s/((1.0+nu_s)*(1.0-2*nu_s)))

I = Identity(2)
F = I + grad((u1-u0)*0.5)
C = F.T*F
E = 0.5*(C-I)
E = variable(E)
W = (lam/2.0)*(tr(E))**2 +mu*tr(E*E)
S = diff(W,E)
P = F*S

Time = 0.5
dt = Constant(0.002)

G1 = rho*dot(p1-p0,v)*dx + dt*inner(P,grad(v))*dx + dot(u1-u0,q)*dx - dt*dot(0.5*(p0+p1),q)*dx
G2 = dt*dot(B,v)*dx #+ dt*dot(T,v)*ds(4) #+ Constant(0)*inner(up1,TestFunction(VV))*dx
G = G1+G2
#G = inner(P,grad(v))*dx - dot(g,v)*dx - dot(T,v)*ds(4)
dt = 0.002
t = dt
while t<= Time:
	#F = I + grad((u1-u0)*0.5)
	solve(G==0,up1,bc1)
	u , p = up1.split()
	plot(u1,mode="displacement")#, interactive=True)
	t += dt
	up0.assign(up1)

plot(U1,mode="displacement", interactive=True)
print coord
print "U1: ", U1(coord)
print "w: ", w_(coord)

print "average", assemble(u[2]*dx)/ assemble(1.0*dx(mesh))
