from mshr import *
from fenics import *
import matplotlib.pyplot as plt

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

a_file = MatReader('data/piececonst_r421_N1024_smooth2.mat')

a_data = a_file.read_field('coeff')

u_result = np.zeros((1024, 421, 421))

class a(UserExpression):
    def __init__(self, file, **kwargs):
        super(a, self).__init__(**kwargs)
        self.data = a_data[file]
    def eval(self, value, x):
        coord_x, coord_y = int(round(x[0]*420)), int(round(x[1]*420))
        value[0] = self.data[coord_x, coord_y]


domain_vertices = [Point(0.5, 0.5),
                    Point(0.5, 0.0),
                    Point(1.0, 0.0),
                    Point(1.0, 1.0),
                    Point(0.0, 1.0),
                    Point(0.0, 0.5),
                    Point(0.5, 0.5)]

domain = Polygon(domain_vertices)

mesh = generate_mesh(domain, 256)

# mesh = UnitSquareMesh(200, 200)

u_D = Constant(0)

V = FunctionSpace(mesh, 'P', 2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

u = Function(V)
v = TestFunction(V)
f = Constant(1.0)

# total_l2 = 0.0

for i in range(1024):
    a_func = a(i)

    F = a_func*dot(grad(u), grad(v))*dx - f*v*dx

    solve(F == 0, u, bc)

    # sol = a_file.read_field('sol')[i]

    # l2 = 0.0

    for x in range(421):
        for y in range(421):
            if x > 210 or y > 210:
            # l2 += (sol[x][y] - u(x/420.0, y/420.0))**2
                u_result[i,x,y] = u(x/420.0, y/420.0)
    
    print(i)

    # total_l2 += l2

np.save('piececonst_r421_N1024_L2', u_result)
# print(total_l2 / 1024)