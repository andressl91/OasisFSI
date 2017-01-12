from dolfin import *
import numpy as np
import sympy as sp
from sympy.printing.ccode import CCodePrinter

def is_sympy(args):
    'Check if args are sympy objects.'
    try:
        return args.__module__.split('.')[0] == 'sympy'
    except AttributeError:
        return False
class DolfinCodePrinter(CCodePrinter):
    '''
    This class provides functionality for converting sympy expression to
    DOLFIN expression. Core of work is done by dolfincode.
    '''
    def __init__(self, settings={}):
        CCodePrinter.__init__(self)

    def _print_Pi(self, expr):
        return 'pi'

def dolfincode(expr, assign_to=None, **settings):
    # Handle scalars
    if is_sympy(expr):
        dolfin_xs = sp.symbols('x[0] x[1] x[2]')
        xs = sp.symbols('x y z')
        for x, dolfin_x in zip(xs, dolfin_xs):
            expr = expr.subs(x, dolfin_x)
        return DolfinCodePrinter(settings).doprint(expr, assign_to)

    # Recurse if vector or tensor
    elif type(expr) is tuple:
        return tuple(dolfincode(e, assign_to, **settings) for e in expr)

def find_my_f_1():
    x,y,t,mu_s,rho_s,lamda = sp.symbols('x y t mu_s rho_s lamda')

    u = x**3 + t**3
    v = y**3 + t**3
    #u = sp.sin(2*sp.pi*t) + sp.cos(sp.pi*y)
    #v = sp.cos(2*sp.pi*t) + sp.cos(sp.pi*x)
    #u = sp.sin(2*sp.pi*t) + sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*y)
    #v = sp.cos(2*sp.pi*t) + sp.cos(2*sp.pi*x)*sp.cos(2*sp.pi*y)

    var = [u,v]

    ux = sp.diff(u,x)
    uy = sp.diff(u,y)
    vx = sp.diff(v,x)
    vy = sp.diff(v,y)

    # x-direction:

    utt = sp.diff(u,t,t) # second derivative
    Fx = ux + 1 + vx
    trE = 0.5*((ux+1)**2+vx**2 - 1 + uy**2+(vy+1)**2-1)
    Ex = 0.5*((ux+1)**2 + (vx)**2 - 1 + (ux+1)*uy+vx*(vy+1))
    sigma_x = lamda*trE + 2*mu_s*Ex
    #F1 =rho_s*utt - sp.diff(sigma_x,x) -sp.diff(sigma_x,y)
    #print sp.simplify(F1)


    # y-direction:
    vtt = sp.diff(v,t,t) # second derivative
    Fy = vy + 1 + uy
    Ey = 0.5*(uy*(ux+1)+(vy+1)*vx + uy**2+(vy+1)**2-1)
    sigma_y = lamda*trE + 2*mu_s*Ey
    F2 = rho_s*vtt - sp.diff(sigma_y,y)
    F1 =rho_s*utt - sp.diff(sigma_x,x)


    #print sp.simplify(F2)
    f = (F1, F2)
    f = dolfincode(f)
    # Expression for the source term in the MMS
    mu_s = 0.5E6
    nu_s = 0.4
    rho_s = 1.0E3
    lamda = nu_s*2*mu_s/(1-2*nu_s)
    t = 0
    f = Expression(f, mu_s=mu_s,lamda=lamda,rho_s=rho_s,t=t)
     # Expression for the velocity components
    u_ = []
    for var_ in var:
        u_.append(dolfincode(var_))

    exact_u = Expression(tuple(u_),t=t)

    return exact_u, f
def find_my_f_2():
    x,y,t,mu_s,rho_s,lamda = sp.symbols('x y t mu_s rho_s lamda')

    u = x**3 + t**3
    v = y**3 + t**3
    #u = sp.sin(2*sp.pi*t) + sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*y)
    #v = sp.cos(2*sp.pi*t) + sp.cos(2*sp.pi*x)*sp.cos(2*sp.pi*y)
    var = [u,v]
    #u = sin(2*pi*t) + sin(2*pi*x)*sin(2*pi*y)
    #v = cos(2*pi*t) + cos(2*pi*x)*cos(2*pi*y)

    ux = sp.diff(u,x)
    uy = sp.diff(u,y)
    vx = sp.diff(v,x)
    vy = sp.diff(v,y)
    uxx = sp.diff(u,x,x)
    uyy = sp.diff(u,y,y)
    vxx = sp.diff(v,x,x)
    vyy = sp.diff(v,y,y)


    # x-direction:

    ut = sp.diff(u,t) # first derivative
    Fx = uxx + 1 + vxx
    trE = 0.5*((uxx+1)**2+vxx**2 - 1 + uyy**2+(vyy+1)**2-1)
    Ex = 0.5*((uxx+1)**2 + (vxx)**2 - 1 + (uxx+1)*uyy+vxx*(vyy+1))
    sigma_x = lamda*trE + 2*mu_s*Ex
    F1 =rho_s*ut+ rho_s*() - sp.diff(sigma_x,x)

    #F1 =rho_s*utt - sp.diff(sigma_x,x) -sp.diff(sigma_x,y)
    #print sp.simplify(F1)


    # y-direction:
    vtt = sp.diff(v,t,t) # second derivative
    Fy = vy + 1 + uy
    Ey = 0.5*(uy*(ux+1)+(vy+1)*vx + uy**2+(vy+1)**2-1)
    sigma_y = lamda*trE + 2*mu_s*Ey
    F2 = rho_s*vtt - sp.diff(sigma_y,y)


    #print sp.simplify(F2)
    f = (F1, F2)
    f = dolfincode(f)
    # Expression for the source term in the MMS
    mu_s = 0.5E6
    nu_s = 0.4
    rho_s = 1.0E3
    lamda = nu_s*2*mu_s/(1 - 2*nu_s)
    t = 0
    f = Expression(f, mu_s=mu_s, lamda=lamda, rho_s=rho_s, t=t)
     # Expression for the velocity components
    u_ = []
    for var_ in var:
        u_.append(dolfincode(var_))

    exact_u = Expression(tuple(u_),t=t)


    return exact_u, f

if __name__ == "__main__":
    exact_u,f= find_my_f()
