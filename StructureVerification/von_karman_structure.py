from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
set_log_active(False)

mesh = Mesh("von_karman_street_FSI_structure.xml")
#print mesh.coordinates()
for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001):
        print coord
        break

#plot(mesh,interactive=True)

V = VectorFunctionSpace(mesh, "CG", 1) # Mesh movement
#VV=V*V
print "Dofs: ",V.dim(), "Cells:", mesh.num_cells()

BarLeftSide =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
BarLeftSide.mark(boundaries,1)
#plot(boundaries,interactive=True)
bc1 = DirichletBC(V, ((0,0)),boundaries, 1)
#bc2 = DirichletBC(VV.sub(1), ((0,0)),boundaries, 1)
bcs = [bc1]#,bc2]

#PARAMETERS:
rho_s = 1.0E3
mu_s = 0.5E6
nu_s = 0.4
E_1 = 1.4E6
lamda = nu_s*2*mu_s/(1-2*nu_s)

#Hookes
def sigma_structure(U):
    return 2*mu_s*sym(grad(U)) + lamda*tr(sym(grad(U)))*Identity(2)
#Second piola stress
def s_s_n_l(U):
    I = Identity(2)
    F = I + grad(U)
    E = 0.5*((F.T*F)-I)
    #J = det(F)
    return lamda*tr(E)*I + 2*mu_s*E



# TEST TRIAL FUNCTIONS

#psi,phi = TestFunctions(VV)
#wU = Function(VV)
#w,U = split(wU)
#U = Function(V)
#U0 = Function(V)
#w0U0 = Function(VV)
#w0,U0 = split(w0U0)
psi = TestFunction(V)
w = Function(V)
w0 = Function(V)
w1 = Function(V)

dt = 0.02
k = Constant(dt)

#Structure Variational form
g = Constant((0,-2*rho_s))
 #Fungerer

#Variational form with double spaces

#G =rho_s*((1./k)*inner(w-w0,psi))*dx + rho_s*inner(dot(grad(0.5*(w+w0)),0.5*(w+w0)),psi)*dx + inner(s_s_n_l(0.5*(U+U0)),grad(psi))*dx \
#- inner(g,psi)*dx - dot(U-U0,phi)*dx + k*dot(0.5*(w+w0),phi)*dx


#Variational form with single space, just solving displacement

G =rho_s*((1./k**2)*inner(w - 2*w0 + w1,psi))*dx + inner(s_s_n_l(0.5*(w+w1)),grad(psi))*dx \
- inner(g,psi)*dx #+ rho_s*inner(dot(grad(W), W),psi)*dx #+ dot(U-U0,phi)*dx - k*dot(0.5*(w+w0),phi)*dx


dis_x = []
dis_y = []

T = 10.0
t = 0
time = np.linspace(0,T,(T/dt)+1)
print "time len",len(time)

from time import sleep
while t < T:
    print "Time: ",t
    ##J = derivative(G,w)
    solve(G == 0, w, bcs, solver_parameters={"newton_solver": \
    {"relative_tolerance": 1E-9,"absolute_tolerance":1E-9,"maximum_iterations":100,"relaxation_parameter":1.0}})

    #w0U0.assign(wU)
    #w,U=split(wU)
    #w,U = wU.split()
    w1.assign(w0)
    w0.assign(w)
    #U0.assign(U)

    #ALE.move(mesh, w)
    #mesh.bounding_box_tree().build(mesh)
    #plot(U,mode="displacement")
    #w.vector()[:] *= float(k)
    #U0.vector()[:] += w.vector()[:]
    #plot(U,mode="displacement")
    #sleep(0.2)
    #print "Ux: %1.4e , Uy: %2.4e "%(U1(coord)[0],U1(coord)[1])
    t += dt
    dis_x.append(w(coord)[0])
    dis_y.append(w(coord)[1])

#print "Ux: %1.4e , Uy: %2.4e "%(U(coord)[0],U(coord)[1])
print len(dis_x), len(time)
plt.plot(time,dis_x,);plt.title("Single space"); plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
plt.show()
plt.plot(time,dis_y);plt.title("Single space");plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
plt.show()
