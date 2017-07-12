from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad

def domain_update(DVP, v_, p_, d_, dvp_,  k, dt, mesh_file, **semimp_namespace):
	print "Second order domain update"
	#Second order extrapolation of fluid-structure interface
	d_n = dvp_["tilde"].sub(0, deepcopy=True)
	d_n1 = dvp_["n-1"].sub(0, deepcopy=True)
	v_n1 = dvp_["n-1"].sub(1, deepcopy=True)
	v_n2 = dvp_["n-2"].sub(1, deepcopy=True)

	d_n.vector().zero()
	d_n.vector().axpy(1, d_n1.vector() \
               + dt*(3./2*v_n1.vector() - 1./2*v_n2.vector()) )

	d = DVP.sub(0).dofmap().collapse(mesh_file)[1].values()
	dvp_["tilde"].vector()[d] = d_n.vector()
	dvp_["n"].vector()[d] = d_n.vector()

	return dict(dvp_=dvp_)
