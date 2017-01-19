from dolfin import *
import vtk
mesh = Mesh("mesh/fluid_new.xml")
V2 = VectorFunctionSpace(mesh,"CG",2)
import matplotlib.pyplot as plt

#d = Function(V2)



"""
t = 0
T = 3
dt = 0.00005
k = Constant(dt)
while t<=T:

    d_.vector()[:] *= float(k) # gives displacement to be used in ALE.move(w_)
    ALE.move(mesh,w_)
    mesh.bounding_box_tree().build(mesh)
    plot(d)
    t +=dt"""
