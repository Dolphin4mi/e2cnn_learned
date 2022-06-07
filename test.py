# from e2cnn.group import SO2
# import os
#
#
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from e2cnn.kernels.irreps_basis import R2ContinuousRotationsSolution
#
# so2 = SO2(3)
#
# r2crs = R2ContinuousRotationsSolution(so2, 4, 5)
# r2crs.sample()
# print(1)






# from e2cnn.group import *
# so2 = SO2(10)
# so2._restrict_irrep(irrep='irrep_1', id=3)
# so2.subgroup(3)
# irrep = so2.irrep(10)
# print(irrep)
#


#
# from e2cnn import gspaces                    1                      #  1
# from e2cnn import nn                                               #  2
# import torch                                                       #  3
#                                                                    #  4
# r2_act = gspaces.Rot2dOnR2(N=8)                                    #  5
# feat_type_in  = nn.FieldType(r2_act,  3*[r2_act.trivial_repr])     #  6
# feat_type_out = nn.FieldType(r2_act, 10*[r2_act.regular_repr])     #  7
#                                                                    #  8
# conv = nn.R2Conv(feat_type_in, feat_type_out, kernel_size=5)       #  9
# relu = nn.ReLU(feat_type_out)                                      # 10
#                                                                    # 11
# x = torch.randn(16, 3, 32, 32)                                     # 12
# x = nn.GeometricTensor(x, feat_type_in)                            # 13
#                                                                    # 14
# y = relu(conv(x))                                                  # 15

# from e2cnn.group import SO2
# # from e2cnn.kernels.basis
# from e2cnn.kernels.irreps_basis import R2ContinuousRotationsSolution
#
# so2 = SO2(3)
# r2crs = R2ContinuousRotationsSolution(so2, 4, 5)
# r2crs.sample()


import e2cnn.nn.init as init
from e2cnn.nn import *
from e2cnn.gspaces import *

import numpy as np
import math

import torch

def test_so2():
    N = 7
    g = Rot2dOnR2(-1, N)

    r1 = FieldType(g, list(g.representations.values()))
    r2 = FieldType(g, list(g.representations.values()))

    s = 7
    # sigma = 0.6
    # fco = lambda r: 1. * r * np.pi
    # fco = lambda r: 2 * r
    sigma = None
    fco = None  # frequencies_cutoff
    cl = R2Conv(r1, r2, s, basisexpansion='blocks',   # convolution layer
                sigma=sigma,
                frequencies_cutoff=fco,
                bias=True)

    for _ in range(8):
        # cl.basisexpansion._init_weights()
        init.generalized_he_init(cl.weights.data, cl.basisexpansion)
        cl.eval()
        cl.check_equivariance()


from e2cnn.group import *
from e2cnn.kernels import *

def test_so2_irreps():

    group = so2_group(10)

    basises = []
    actions = []
    for in_rep in group.irreps.values():
        for out_rep in group.irreps.values():
            basis = kernels_SO2_act_R2(in_rep, out_rep,
                                       radii=[0., 1., 2., 5, 10],
                                       sigma=[0.6, 1., 1.3, 2.5, 3.]
                                       )
            basises.appenSd(basis)
            action = group.irrep(1)
            actions.append(action)
            # self._check(basis, group, in_rep, out_rep, action)
    a = basises[50].angular.irreps_bases
    b = a[('irrep_4', 'irrep_6')]
    sam = np.array([[1, 2, 3]])
    c = b.sample(sam)
    print(1)


if __name__=="__main__":
    test_so2()
    # test_so2_irreps()

