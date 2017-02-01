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

def find_my_f():
    x, y, p, wx, wy, u, v, t, mu_f, rho_f = sp.symbols('x y p wx wy u v t mu_f rho_f')

    dx = sp.sin(t)*(x-2)
    dy = 0.0
    u = sp.cos(t)*(x-2)
    v = sp.cos(t)*(y-2)
    wx = sp.cos(t)*(x-2)
    wy = 0.0
    var = [u,v]
    w_var = [wx, wy]

    ux = sp.diff(u,x)
    uy = sp.diff(u,y)
    vx = sp.diff(v,x)
    vy = sp.diff(v,y)
    uxx = sp.diff(ux,x)
    uxy = sp.diff(ux,y)
    uyx = sp.diff(uy,x)
    uyy = sp.diff(uy,y)
    vxx = sp.diff(vx,x)
    vxy = sp.diff(vx,y)
    vyx = sp.diff(vy,x)
    vyy = sp.diff(vy,y)
    px = sp.diff(p,x)
    py = sp.diff(p,y)
    # x-direction:
    ut = sp.diff(u,t) # first derivative

    sigma_x = -px + mu_f*(uxx + vxy + vyx + vxy)

    F1 =rho_f*ut - sigma_x

    #F1 =rho_s*utt - sp.diff(sigma_x,x) -sp.diff(sigma_x,y)
    #print sp.simplify(F1)


    # y-direction:
    vt = sp.diff(v,t) # second derivative
    sigma_y = -py + mu_f*(uyx + vyy + uyx + uxy)
    F2 = rho_f*vt -sigma_y


    #print sp.simplify(F2)
    f = (F1, F2)
    f = dolfincode(f)
    # Expression for the source term in the MMS
    mu_f = 1
    nu_f = 1.0E-3
    rho_f = 1.0E3
    lamda = nu_f*2*mu_f/(1-2*nu_f)
    t = 0
    f = Expression(f, mu_f=mu_f,lamda=lamda,rho_f=rho_f,t=t)
     # Expression for the velocity components
    u_ = []
    w_ = []
    for var_ in var:
        u_.append(dolfincode(var_))
    for w_var_ in w_var:
        w_.append(dolfincode(w_var_))
    exact_u = Expression(tuple(u_),t=t)
    #exact_w = Expression(tuple(w_),t=t)


    return exact_u,f
"""def find_my_ufl_F():
    #x,y,p,wx,wy,t,mu_f,rho_f = sp.symbols('x y p wx wy t mu_f rho_f')


    # Making of F with ufl:
    w_ufl = sin(u**2)
    d_ufl = np.sin(t)*(x-2)
    dy = 0
    wx = np.cos(t)*(x-2)
    wy = 0


    u = Function(V1)
    u_ex = np.cos(t)*(u-2)
    v_ex = np.cos(t)*(u-2)
    # Annotate expression w as a variable that can be used in diff
    u_ex = variable(u_ex)
    dudx = diff(u,x)

    t = variable(Constant(cell))

    dudt = diff(u, t)
    return exact_u,f



    return exact_u,f"""
if __name__ == "__main__":
    exact_u, exact_w, f= find_my_f()
