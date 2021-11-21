from mshr import *
from fenics import *
import matplotlib.pyplot as plt
import torch

import scipy.io
import h5py

import numpy as np

class MatReader(object):
    def __init__(self, file_path, to_torch=False, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path, 'r')
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

a_file = MatReader('data/piececonst_r421_N1024_smooth1.mat')

# plt.subplot(1,4,1)

a_data = a_file.read_field('coeff')[0]

# plt.imshow(a_data)

# def a(x, y):
#     x, y = round(x * 421 / 2.0), round(y * 421 / 2.0)
#     return a_data[x,y]

class a(UserExpression):
    def eval(self, value, x):
        coord_x, coord_y = int(round(x[0]*420)), int(round(x[1]*420))
        value[0] = a_data[coord_x, coord_y]


domain_vertices = [Point(0.5, 0.5),
                    Point(0.5, 0.0),
                    Point(1.0, 0.0),
                    Point(1.0, 1.0),
                    Point(0.0, 1.0),
                    Point(0.0, 0.5),
                    Point(0.5, 0.5)]

# domain_vertices = [Point(0.0, 0.0),
#                     Point(1.0, 0.0),
#                     Point(1.0, 1.0),
#                     Point(0.0, 1.0),
#                     Point(0.0, 0.0)]

domain = Polygon(domain_vertices)

mesh = generate_mesh(domain, 16)
# mesh = UnitSquareMesh(200, 200)
# PolygonalMeshGenerator.generate(mesh, domain_vertices, 0.25)

# plt.subplot(1,4,2)

a_plot = a()

# plot(a_plot, mesh = mesh)

# plt.subplot(1,4,3)

# plot(mesh)

u_D = Constant(0)

V = FunctionSpace(mesh, 'P', 2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

u = Function(V)
v = TestFunction(V)
f = Constant(1.0)

a_func = a()

F = a_func*dot(grad(u), grad(v))*dx - f*v*dx

solve(F == 0, u, bc)

plot(u)
plot(mesh)

# print(u(0.1,0.1))

# plt.subplot(1,4,4)

sol = a_file.read_field('sol')[0]

# plt.imshow(sol)

total_l2 = 0.0

for x in range(421):
    for y in range(421):
        if x > 210 or y > 210:
            total_l2 += (sol[x][y] - u(x/420.0, y/420.0))**2

print(total_l2)

plt.show()
plt.savefig('darcy_mesh.png')