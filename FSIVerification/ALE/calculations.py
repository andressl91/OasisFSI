from dolfin import *
from mapping import *

def integrateFluidStress(u,p,d,mu_f,n,ds_s,dS_s):
    Dr = -assemble((sigma_f_new(u,p,d,mu_f)*n)[0]*ds_s)
    Li = -assemble((sigma_f_new(u,p,d,mu_f)*n)[1]*ds_s)
    Dr += -assemble((sigma_f_new(u('-'),p('-'),d('-'),mu_f)*n('-'))[0]*dS_s)
    Li += -assemble((sigma_f_new(u('-'),p('-'),d('-'),mu_f)*n('-'))[1]*dS_s)
    return Dr,Li
