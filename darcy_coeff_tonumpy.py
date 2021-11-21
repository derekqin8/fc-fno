from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.autograd import Variable
import torch.autograd

import operator
from functools import reduce
from functools import partial

import numpy as np
import scipy.io
import h5py

import time
import matplotlib.pyplot as plt

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
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


train_file = MatReader('darcy_r421_Lshape/piececonst_r421_N1024_smooth1.mat')

x_train = train_file.read_field('coeff').numpy()

np.save("darcy_r421_Lshape/piececonst_r421_N1024_coeff1.npy", x_train)

test_file = MatReader('darcy_r421_Lshape/piececonst_r421_N1024_smooth2.mat')

x_test = test_file.read_field('coeff').numpy()

np.save("darcy_r421_Lshape/piececonst_r421_N1024_coeff2.npy", x_test)