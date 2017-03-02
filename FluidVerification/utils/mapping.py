from fenics import Identity, det, grad

#Deformation gradient
def F_(U):
	I = Identity(U.ufl_shape[0])
	return (I + grad(U))

#Jacobian of Deformation Gradient
def J_(U):
	return det(F_(U))
