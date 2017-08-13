from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad

def domain_update(VPD, vpd_, d_, dt, mesh_file, **semimp_namespace):
    print "Second order domain update"
    #Second order extrapolation of fluid-structure interface
    d_n =  vpd_["tilde"].sub(2, deepcopy=True)
    d_n1 = vpd_["n-1"].sub(2, deepcopy=True)
    d_n2 = vpd_["n-2"].sub(2, deepcopy=True)
    d_n3 = vpd_["n-3"].sub(2, deepcopy=True)

    d_n.vector().zero()
    d_n.vector().axpy(1, 5./2.*d_n1.vector() - 2*d_n2.vector() + 1./2*d_n3.vector())

    d = VPD.sub(2).dofmap().collapse(mesh_file)[1].values()
    vpd_["tilde"].vector()[d] = d_n.vector()
    vpd_["n"].vector()[d] = d_n.vector()

    return {} #dict(vpd_=dw_)
