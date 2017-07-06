from dolfin import *

def solver_setup(F_correction, F_tentative, fluid_sol, solid_sol, **monolithic):


    a = lhs(F_correction)
    A_corr = assemble(a, keep_diagonal=True)
    fluid_sol.set_operator(A_corr)

    a = lhs(F_tentative)
    A_tent = assemble(a, keep_diagonal=True)
    #A_tent.ident_zeros()


    return dict(A_corr=A_corr, A_tent=A_tent, fluid_sol=fluid_sol)

def Fluid_extrapolation(F_extrapolate, DW, dw_, vp_, bcs_w, mesh_file, \
                        k, **semimp_namespace):
    print "No extrapolation"

    return {}

def Fluid_tentative(F_tentative, A_tent, VP, vp_, bcs_tent, \
     mesh_file, **semimp_namespace):


    L = rhs(F_tentative)
    b = assemble(L)
    [bc.apply(A_tent, b) for bc in bcs_tent]

    print "Solving tentative velocity"
    solve(A_tent, vp_["tilde"].vector(), b)

    #Convert solution into mixedspacefunction
    #v = VP.sub(0).dofmap().collapse(mesh_file)[1].values()
    #vp_["tilde"].vector()[v] = vp_["tilde"].sub(0, deepcopy=True).vector()

    return dict(vp_=vp_)


def Fluid_correction(mesh_file, VP, A_corr, F_correction, bcs_corr, \
                vp_, dw_, fluid_sol, T, t, **monolithic):

    b = assemble(rhs(F_correction))
    [bc.apply(A_corr, b) for bc in bcs_corr]

    print "Solving correction velocity"
    solve(A_corr, vp_["n"].vector(), b)
    return dict(t=t, vp_=vp_)


def Solid_momentum(F_solid, Jac_solid, bcs_solid, vp_, \
                DW, dw_, solid_sol, dw_res, rtol, atol, max_it, T, t, **monolithic):

    print "Dummy solid"
    return {}
