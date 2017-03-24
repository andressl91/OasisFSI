from fenics import *

def fluid_tentative(N, v_deg, p_deg, T, dt, rho, mu, **problem_namespace):



    class inlet(Expression):
    	def __init__(self):
    		self.t = 0
    	def eval(self,value,x):
    		value[0] = 0.5*(1-np.cos(self.t*np.pi/2))*1.5*U_in*x[1]*(H-x[1])/((H/2.0)**2)
    		value[1] = 0
    	def value_shape(self):
    		return (2,)

    inlet = inlet()
    #Fluid velocity conditions
    u_inlet  = DirichletBC(VQ.sub(0), inlet, boundaries, 3)
    u_wall   = DirichletBC(VQ.sub(0), ((0.0, 0.0)), boundaries, 2)
    u_circ   = DirichletBC(VQ.sub(0), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
    u_bar    = DirichletBC(VQ.sub(0), ((0.0, 0.0)), boundaries, 5) #No slip on geometry in fluid
    u_barwall= DirichletBC(VQ.sub(0), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

    #Pressure Conditions
    p_out = DirichletBC(VVQ.sub(2), 0, boundaries, 4)

    bc_up = [u_inlet, u_wall, u_circ, u_barwall, p_out]

    #Tilde
    F_tilde = (rho/k)*J_(d_vec)*inner((u_t - u_["n-1"]), psi)*dx(1)
    F_tilde += rho*J_(d_vec)*inner(inv(F_(d_vec))*dot(grad(u_t), u0_tilde - w_vec), psi)*dx
    F_tilde += J_(d_vec)*mu*inner(grad(u_t)*inv(F_(d_vec)), grad(psi)*inv(F_(d_vec)) ) *dx
    #F1 += J_(d_vec)*inner(mu*sigma_f_shearstress_map(u_t, d_vec)*inv(F_(d_vec)).T, sigma_f_shearstress_map(psi, d_vec))*dx


    # Pressure update
    F_corr = rho/k*J_(d_vec)*inner(u_["n"] - u_tilde, v)*dx
    #F_corr += rho*inner(J_(d_vec)* -inv(F_(d_vec))*dot(grad(u_["n"]), w_vec), v)*dx
    #F2 += J_(d_vec)*inner(inv(F_(d_vec)).T*grad(p_["n"]), v)*dx
    F_corr -= p_["n"]*J_(d_vec)*inner(inv(F_(d_vec)).T, grad(v))*dx
    #F2 -= inner(J_(d_vec)*p_["n"], div(v))*dx

    F_corr += J_(d_vec)*inner(grad(u_["n"]), inv(F_(d_vec)).T)*q*dx
    #F2 += inner(q, div(J_(d_vec)*inv(F_(d_vec))*u_["n"]))*dx
    #F2 += J_(d_vec)*inner(dot(inv(F_(d_vec))*u_["n"], n) \
    #    - dot(inv(F_(d_vec))*w_vec, n), dot(v, n))*ds
