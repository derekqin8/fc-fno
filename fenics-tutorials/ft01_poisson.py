from fenics import *

# defines a unit square mesh divided into an 8x8 grid
mesh = UnitSquareMesh(8,8)

# Create a finite element function space V
# 'P' = 'Lagrange': the Lagrange family of elements
# 1 for P_1 element, aka linearly compute x,y over the element
# Family of finite elements here: https://www-users.cse.umn.edu/~arnold/femtable/
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree = 2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define trial and test functions over the space
# Trial function: target function of the PDE
# Test function: used for integration by parts, can be any function over the domain
u = TrialFunction(V)
v = TestFunction(V)

f = Constant(-6.0)
a = dot(grad(u), grad(v)) * dx
L = f*v*dx

# Compute solution
# u is a function on space V
u = Function(V)

# Solve a = L equation for equation u with bc boundary condition
solve(a == L, u, bc)

# Compute error
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

import matplotlib.pyplot as plt

# Plot solution and mesh
plot(u)
plot(mesh)

# Hold plot
plt.show()
plt.savefig('ft01_poisson.png')