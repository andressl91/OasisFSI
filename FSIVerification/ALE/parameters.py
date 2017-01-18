#BOUNDARY CONDITIONS
# FLUID
#FSI = FSI_deg
nu = 10**-3
rho_f = 1.0*1e3
mu_f = rho_f*nu

# SOLID
Pr = 0.4
H = 0.41
L = 2.5

def U_in(FSI):
    return [0.2, 1.0, 2.0][FSI-1]   # Reynolds vel
def mu_s(FSI):
    return [0.5, 0.5, 2.0][FSI-1]*1e6
def rho_s(FSI):
    return [1.0, 10, 1.0][FSI-1]*1e3
def lamda_s(FSI,mu):
    return 2*mu*Pr/(1-2.*Pr)
