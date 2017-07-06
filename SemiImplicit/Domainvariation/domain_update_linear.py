from dolfin import Constant, inner, inv, dot, grad, det, Identity,\
solve, lhs, rhs, assemble, DirichletBC, div, sym, nabla_grad

def domain_update(DW, w_, d_, dw_,  k, dt, mesh_file, **semimp_namespace):
	print "Second order domain update"
	#Second order extrapolation of fluid-structure interface
	d_n =  dw_["tilde"].sub(0, deepcopy=True)
	d_n1 = dw_["n-1"].sub(0, deepcopy=True)
	w_n1 = dw_["n-1"].sub(1, deepcopy=True)
	w_n2 = dw_["n-2"].sub(1, deepcopy=True)

	d_n.vector().zero()
	d_n.vector().axpy(1, d_n1.vector() \
	            + dt*(3./2*w_n1.vector() - 1./2*w_n2.vector()))

	d = DW.sub(0).dofmap().collapse(mesh_file)[1].values()
	dw_["tilde"].vector()[d] = d_n.vector()
	dw_["n"].vector()[d] = d_n.vector()

	return dict(dw_=dw_)
