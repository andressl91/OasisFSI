from sympy import *

x,y,t,rho_s, mu_s,lamda = symbols('x[0] x[1] t rho_s mu_s lamda')

u = x**2 + t**2
v = y**2 + t**2

ux = diff(u,x)
uy = diff(u,y)
vx = diff(v,x)
vy = diff(v,y)

# x-direction:

utt = diff(u,t,t) # second derivative
Fx = ux + 1 + vx
trE = (ux+1)**2+vx**2-1 + uy**2+(vy+1)**2-1
Ex = (ux+1)**2 + (vx)**2 -1 + (ux+1)*uy+vy*(vy+1)
sigma_x = Fx*lamda*trE + 2*mu_s*Ex
F1 =utt- diff(sigma_x,x)
print simplify(F1)


# y-direction:
vtt = diff(v,t,t) # second derivative
Fy = vy + 1 + uy
Ey = uy*(ux+1)+(vy+1)*vx + uy**2+(vy+1)**2-1
sigma_y = Fy*lamda*trE + 2*mu_s*Ey
F2 = vtt - diff(sigma_y,y)
print simplify(F2)
