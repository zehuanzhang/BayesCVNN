"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import torch
import torch.nn as nn
from Src.Net_Parts_Torch import Complex_DoubleConv, Complex_Down_DoubleConv2d, Complex_Up, Complex_OutConv, Complex_DoubleConv_RI, Complex_Down_DoubleConv2d_RI
from Src.Net_Parts_Torch import Complex_Linear, Complex_Flatten
from Src.ComplexValuedAutoencoder_Class_Torch import ComplexBatchNorm2d, Complexavg_pool2d, ComplexReLU, ComplexConv2d, Complexavg_pool2d, ComplexConv2d_RI
from complex_dropout import Complex_BernoulliDropout


class end_to_end_Net_classification_sparsity_RI(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_classes, bilinear=False, sparsity_r=0.0, sparsity_i=0.0):
        super(end_to_end_Net_classification_sparsity_RI, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sparsity_r = sparsity_r
        self.sparsity_i = sparsity_i
        self.save_r = 1 - sparsity_r
        self.save_i = 1 - sparsity_i

        ############## Autoencoder

        self.inc1 = Complex_DoubleConv_RI(in_channels=n_in_channels, out_channels_r=int(16*self.save_r), out_channels_i=int(16*self.save_i), mid_channels=None, kernel_size=3)
        self.down1 = Complex_Down_DoubleConv2d_RI(in_channels=max(int(16*self.save_r), int(16*self.save_i)), out_channels_r=int(32*self.save_r), out_channels_i=int(32*self.save_i), Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down2 = Complex_Down_DoubleConv2d_RI(in_channels=max(int(32*self.save_r), int(32*self.save_i)), out_channels_r=int(64*self.save_r), out_channels_i=int(64*self.save_i), Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down3 = Complex_Down_DoubleConv2d_RI(in_channels=max(int(64*self.save_r), int(64*self.save_i)), out_channels_r=int(128*self.save_r), out_channels_i=int(128*self.save_i), Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        factor = 2 if bilinear else 1
        self.down4 = Complex_Down_DoubleConv2d_RI(in_channels=max(int(128*self.save_r), int(128*self.save_i)), out_channels_r=int(256 // factor*self.save_r), out_channels_i=int(256 // factor*self.save_i), Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)

        self.feature = Complex_DoubleConv_RI(in_channels=max(int(256 // factor*self.save_r), int(256 // factor*self.save_i)), out_channels_r=int(256 // factor*self.save_r), out_channels_i=int(256 // factor*self.save_i), mid_channels=None,
                                          kernel_size=3)



        ############### Classifier

        self.conv1 = ComplexConv2d_RI(in_channels=max(int(256 // factor*self.save_r), int(256 // factor*self.save_i)), out_channels_r=int(200*self.save_r), out_channels_i=int(200*self.save_i), kernel_size=3, stride=1, padding='same',
                      dilation=1, groups=1, bias=True)
        self.ReLU1 = ComplexReLU()
        self.pool1 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0,
                              dilation=1, return_indices=False, ceil_mode=False,
                              count_include_pad=True, divisor_override=None)
        self.conv2 = ComplexConv2d_RI(in_channels=max(int(200*self.save_r), int(200*self.save_i)), out_channels_r=int(256*self.save_r), out_channels_i=int(256*self.save_i), kernel_size=3, stride=1, padding='same',
                                     dilation=1, groups=1, bias=True)
        self.ReLU2 = ComplexReLU()
        self.flat1 = Complex_Flatten()
        self.lin1 = Complex_Linear(in_channels=576, out_channels=2000, activation="relu")#2070 1152 2304       1.5:918  684 576   0.5:2070 1836  1728
        self.lin2 = Complex_Linear(in_channels=2000, out_channels=1000, activation="relu")
        self.lin3 = Complex_Linear(in_channels=1000, out_channels=500, activation="relu")
        self.lin4 = Complex_Linear(in_channels=500, out_channels=100, activation="relu")
        self.out1 = Complex_Linear(in_channels=100, out_channels=n_classes, activation="None")



    def forward(self, x):
        # print('forward in net')

        i1 = self.inc1(x)
        d1 = self.down1(i1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        f = self.feature(d4)

        c1 = self.ReLU1(self.conv1(f))
        c2 = self.pool1(c1)
        c3 = self.ReLU2(self.conv2(c2))
        c4 = self.flat1(c3)
        # print(c4.shape)
        c5 = self.lin1(c4)
        c6 = self.lin2(c5)
        c7 = self.lin3(c6)
        c8 = self.lin4(c7)
        classified = abs(self.out1(c8))

        return classified




########################################################################################################################

class end_to_end_Net_classification_sparsity_RI_lastlayer(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_classes, bilinear=False, sparsity_r=0.0, sparsity_i=0.0):
        super(end_to_end_Net_classification_sparsity_RI_lastlayer, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.sparsity_r = sparsity_r
        self.sparsity_i = sparsity_i
        self.save_r = 1 - sparsity_r
        self.save_i = 1 - sparsity_i

        ############## Autoencoder

        self.inc1 = Complex_DoubleConv(in_channels=n_in_channels, out_channels=16, mid_channels=None, kernel_size=3)
        self.down1 = Complex_Down_DoubleConv2d(in_channels=16, out_channels=32, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down2 = Complex_Down_DoubleConv2d(in_channels=32, out_channels=64, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down3 = Complex_Down_DoubleConv2d(in_channels=64, out_channels=128, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        factor = 2 if bilinear else 1
        self.down4 = Complex_Down_DoubleConv2d(in_channels=128, out_channels=256 // factor, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)

        self.feature = Complex_DoubleConv(in_channels=256 // factor, out_channels=256 // factor, mid_channels=None,
                                          kernel_size=3)



        ############### Classifier

        self.conv1 = ComplexConv2d(in_channels=128, out_channels=200, kernel_size=3, stride=1, padding='same',
                      dilation=1, groups=1, bias=True)
        self.ReLU1 = ComplexReLU()
        self.pool1 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0,
                              dilation=1, return_indices=False, ceil_mode=False,
                              count_include_pad=True, divisor_override=None)
        # self.conv2 = ComplexConv2d(in_channels=200, out_channels=256, kernel_size=3, stride=1, padding='same',
        #                              dilation=1, groups=1, bias=True)
        self.conv2 = ComplexConv2d_RI(in_channels=200,
                                      out_channels_r=int(256 * self.save_r), out_channels_i=int(256 * self.save_i),
                                      # out_channels_r=256, out_channels_i=256,
                                      kernel_size=3, stride=1, padding='same',
                                      dilation=1, groups=1, bias=True)
        self.ReLU2 = ComplexReLU()
        self.flat1 = Complex_Flatten()
        self.lin1 = Complex_Linear(in_channels=2070, out_channels=2000, activation="relu")
        self.lin2 = Complex_Linear(in_channels=2000, out_channels=1000, activation="relu")
        self.lin3 = Complex_Linear(in_channels=1000, out_channels=500, activation="relu")
        self.lin4 = Complex_Linear(in_channels=500, out_channels=100, activation="relu")
        self.out1 = Complex_Linear(in_channels=100, out_channels=n_classes, activation="None")



    def forward(self, x):

        i1 = self.inc1(x)
        d1 = self.down1(i1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        f = self.feature(d4)

        c1 = self.ReLU1(self.conv1(f))
        c2 = self.pool1(c1)
        c3 = self.ReLU2(self.conv2(c2))
        c4 = self.flat1(c3)
        c5 = self.lin1(c4)
        c6 = self.lin2(c5)
        c7 = self.lin3(c6)
        c8 = self.lin4(c7)
        classified = abs(self.out1(c8))

        return classified


########################################################################################################################




class end_to_end_Net_classification_sparsity(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_classes, bilinear=False, sparsity=0.5):
        super(end_to_end_Net_classification_sparsity, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sparsity = sparsity
        self.save = 1 - sparsity

        ############## Autoencoder

        self.inc1 = Complex_DoubleConv(in_channels=n_in_channels, out_channels=int(16*self.save), mid_channels=None, kernel_size=3)
        self.down1 = Complex_Down_DoubleConv2d(in_channels=int(16*self.save), out_channels=int(32*self.save), Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down2 = Complex_Down_DoubleConv2d(in_channels=int(32*self.save), out_channels=int(64*self.save), Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down3 = Complex_Down_DoubleConv2d(in_channels=int(64*self.save), out_channels=int(128*self.save), Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        factor = 2 if bilinear else 1
        self.down4 = Complex_Down_DoubleConv2d(in_channels=int(128*self.save), out_channels=int(256 // factor*self.save), Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)

        self.feature = Complex_DoubleConv(in_channels=int(256 // factor*self.save), out_channels=int(256 // factor*self.save), mid_channels=None,
                                          kernel_size=3)



        ############### Classifier

        self.conv1 = ComplexConv2d(in_channels=int(128*self.save), out_channels=int(200*self.save), kernel_size=3, stride=1, padding='same',
                      dilation=1, groups=1, bias=True)
        self.ReLU1 = ComplexReLU()
        self.pool1 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0,
                              dilation=1, return_indices=False, ceil_mode=False,
                              count_include_pad=True, divisor_override=None)
        self.conv2 = ComplexConv2d(in_channels=int(200*self.save), out_channels=int(256*self.save), kernel_size=3, stride=1, padding='same',
                                     dilation=1, groups=1, bias=True)
        self.ReLU2 = ComplexReLU()
        self.flat1 = Complex_Flatten()
        self.lin1 = Complex_Linear(in_channels=1152, out_channels=2000, activation="relu")#1152 2304
        self.lin2 = Complex_Linear(in_channels=2000, out_channels=1000, activation="relu")
        self.lin3 = Complex_Linear(in_channels=1000, out_channels=500, activation="relu")
        self.lin4 = Complex_Linear(in_channels=500, out_channels=100, activation="relu")
        self.out1 = Complex_Linear(in_channels=100, out_channels=n_classes, activation="None")



    def forward(self, x):

        i1 = self.inc1(x)
        d1 = self.down1(i1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        f = self.feature(d4)

        c1 = self.ReLU1(self.conv1(f))
        c2 = self.pool1(c1)
        c3 = self.ReLU2(self.conv2(c2))
        c4 = self.flat1(c3)
        # print(c4.shape)
        c5 = self.lin1(c4)
        c6 = self.lin2(c5)
        c7 = self.lin3(c6)
        c8 = self.lin4(c7)
        classified = abs(self.out1(c8))

        return classified









class end_to_end_Net_classification_small_Bayesian(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_classes, layer_setting, dropout_rate=0.25):
        super(end_to_end_Net_classification_small_Bayesian, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_classes = n_classes
        self.kernel_size = 3

        ############## Autoencoder

        self.conv1 = ComplexConv2d(in_channels=self.n_in_channels, out_channels=16, kernel_size=self.kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn1 = ComplexBatchNorm2d(num_features=16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.ReLU1 = ComplexReLU()
        self.pool1 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, count_include_pad=True, divisor_override=None)
        # self.dropout1 = Complex_BernoulliDropout(dropout_rate, real_imag=layer_setting[0])

        self.conv2 = ComplexConv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn2 = ComplexBatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.ReLU2 = ComplexReLU()
        self.pool2 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.dropout2 = Complex_BernoulliDropout(dropout_rate, real_imag=layer_setting[0])

        self.conv3 = ComplexConv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn3 = ComplexBatchNorm2d(num_features=64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.ReLU3 = ComplexReLU()
        self.pool3 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.dropout3 = Complex_BernoulliDropout(dropout_rate, real_imag=layer_setting[1])

        self.conv4 = ComplexConv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn4 = ComplexBatchNorm2d(num_features=128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.ReLU4 = ComplexReLU()
        self.pool4 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.dropout4 = Complex_BernoulliDropout(dropout_rate, real_imag=layer_setting[2])

        ############### Classifier

        self.conv5 = ComplexConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same', dilation=1, groups=1, bias=True)
        self.ReLU5 = ComplexReLU()

        self.flat1 = Complex_Flatten()
        self.lin1 = Complex_Linear(in_channels=4608, out_channels=2000, activation="relu")
        self.lin2 = Complex_Linear(in_channels=2000, out_channels=1000, activation="relu")
        self.lin3 = Complex_Linear(in_channels=1000, out_channels=500, activation="relu")
        self.lin4 = Complex_Linear(in_channels=500, out_channels=100, activation="relu")
        self.out1 = Complex_Linear(in_channels=100, out_channels=n_classes, activation="None")





    def forward(self, x):

        c1 = self.ReLU1(self.bn1(self.conv1(x)))
        # print(c1.shape)
        # p1 = self.dropout1(self.pool1(c1))
        p1 = self.pool1(c1)
        # print(p1.shape)

        c2 = self.ReLU2(self.bn2(self.conv2(p1)))
        # print(c2.shape)
        p2 = self.dropout2(self.pool2(c2))
        # print(p2.shape)

        c3 = self.ReLU3(self.bn3(self.conv3(p2)))
        # print(c3.shape)
        p3 = self.dropout3(self.pool3(c3))
        # print(p3.shape)

        c4 = self.ReLU4(self.bn4(self.conv4(p3)))
        # print(c4.shape)
        p4 = self.dropout4(self.pool4(c4))
        # print(p4.shape)

        c5 = self.ReLU5(self.conv5(p4))
        # print(c5.shape)


        c6 = self.flat1(c5)
        l1 = self.lin1(c6)
        l2 = self.lin2(l1)
        l3 = self.lin3(l2)
        l4 = self.lin4(l3)
        classified = abs(self.out1(l4))

        return classified










class end_to_end_Net_classification_small(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_classes, bilinear=False):
        super(end_to_end_Net_classification_small, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_classes = n_classes
        self.kernel_size = 3

        ############## Autoencoder

        self.conv1 = ComplexConv2d(in_channels=self.n_in_channels, out_channels=16, kernel_size=self.kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn1 = ComplexBatchNorm2d(num_features=16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.ReLU1 = ComplexReLU()
        self.pool1 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, count_include_pad=True, divisor_override=None)

        self.conv2 = ComplexConv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn2 = ComplexBatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.ReLU2 = ComplexReLU()
        self.pool2 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, count_include_pad=True, divisor_override=None)

        self.conv3 = ComplexConv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn3 = ComplexBatchNorm2d(num_features=64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.ReLU3 = ComplexReLU()
        self.pool3 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, count_include_pad=True, divisor_override=None)

        self.conv4 = ComplexConv2d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn4 = ComplexBatchNorm2d(num_features=128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.ReLU4 = ComplexReLU()
        self.pool4 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, count_include_pad=True, divisor_override=None)

        ############### Classifier

        self.conv5 = ComplexConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same', dilation=1, groups=1, bias=True)
        self.ReLU5 = ComplexReLU()

        self.flat1 = Complex_Flatten()
        self.lin1 = Complex_Linear(in_channels=4608, out_channels=2000, activation="relu")
        self.lin2 = Complex_Linear(in_channels=2000, out_channels=1000, activation="relu")
        self.lin3 = Complex_Linear(in_channels=1000, out_channels=500, activation="relu")
        self.lin4 = Complex_Linear(in_channels=500, out_channels=100, activation="relu")
        self.out1 = Complex_Linear(in_channels=100, out_channels=n_classes, activation="None")





    def forward(self, x):

        c1 = self.ReLU1(self.bn1(self.conv1(x)))
        # print(c1.shape)
        p1 = self.pool1(c1)
        # print(p1.shape)

        c2 = self.ReLU2(self.bn2(self.conv2(p1)))
        # print(c2.shape)
        p2 = self.pool2(c2)
        # print(p2.shape)

        c3 = self.ReLU3(self.bn3(self.conv3(p2)))
        # print(c3.shape)
        p3 = self.pool3(c3)
        # print(p3.shape)

        c4 = self.ReLU4(self.bn4(self.conv4(p3)))
        # print(c4.shape)
        p4 = self.pool4(c4)
        # print(p4.shape)

        c5 = self.ReLU5(self.conv5(p4))
        # print(c5.shape)


        c6 = self.flat1(c5)
        l1 = self.lin1(c6)
        l2 = self.lin2(l1)
        l3 = self.lin3(l2)
        l4 = self.lin4(l3)
        classified = abs(self.out1(l4))

        return classified


class end_to_end_Net_classification(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_classes, bilinear=False):
        super(end_to_end_Net_classification, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        ############## Autoencoder

        self.inc1 = Complex_DoubleConv(in_channels=n_in_channels, out_channels=16, mid_channels=None, kernel_size=3)
        self.down1 = Complex_Down_DoubleConv2d(in_channels=16, out_channels=32, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down2 = Complex_Down_DoubleConv2d(in_channels=32, out_channels=64, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down3 = Complex_Down_DoubleConv2d(in_channels=64, out_channels=128, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        factor = 2 if bilinear else 1
        self.down4 = Complex_Down_DoubleConv2d(in_channels=128, out_channels=256 // factor, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)

        self.feature = Complex_DoubleConv(in_channels=256 // factor, out_channels=256 // factor, mid_channels=None,
                                          kernel_size=3)



        ############### Classifier

        self.conv1 = ComplexConv2d(in_channels=128, out_channels=200, kernel_size=3, stride=1, padding='same',
                      dilation=1, groups=1, bias=True)
        self.ReLU1 = ComplexReLU()
        self.pool1 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0,
                              dilation=1, return_indices=False, ceil_mode=False,
                              count_include_pad=True, divisor_override=None)
        self.conv2 = ComplexConv2d(in_channels=200, out_channels=256, kernel_size=3, stride=1, padding='same',
                                     dilation=1, groups=1, bias=True)
        self.ReLU2 = ComplexReLU()
        self.flat1 = Complex_Flatten()
        self.lin1 = Complex_Linear(in_channels=2304, out_channels=2000, activation="relu")
        self.lin2 = Complex_Linear(in_channels=2000, out_channels=1000, activation="relu")
        self.lin3 = Complex_Linear(in_channels=1000, out_channels=500, activation="relu")
        self.lin4 = Complex_Linear(in_channels=500, out_channels=100, activation="relu")
        self.out1 = Complex_Linear(in_channels=100, out_channels=n_classes, activation="None")



    def forward(self, x):

        i1 = self.inc1(x)
        d1 = self.down1(i1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        f = self.feature(d4)
        print(f.shape)

        c1 = self.ReLU1(self.conv1(f))
        c2 = self.pool1(c1)
        c3 = self.ReLU2(self.conv2(c2))
        print(c3.shape)
        c4 = self.flat1(c3)
        c5 = self.lin1(c4)
        c6 = self.lin2(c5)
        c7 = self.lin3(c6)
        c8 = self.lin4(c7)
        classified = abs(self.out1(c8))

        return classified



class end_to_end_Net(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_classes, bilinear=False,
                 feature_extractor=False):
        super(end_to_end_Net, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_extractor = feature_extractor

        ############## Autoencoder

        self.inc1 = Complex_DoubleConv(in_channels=n_in_channels, out_channels=16, mid_channels=None, kernel_size=3)
        self.down1 = Complex_Down_DoubleConv2d(in_channels=16, out_channels=32, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down2 = Complex_Down_DoubleConv2d(in_channels=32, out_channels=64, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        self.down3 = Complex_Down_DoubleConv2d(in_channels=64, out_channels=128, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)
        factor = 2 if bilinear else 1
        self.down4 = Complex_Down_DoubleConv2d(in_channels=128, out_channels=256 // factor, Conv2d_kernel_size=3,
                                               Pooling_kernel_size=2)

        self.feature = Complex_DoubleConv(in_channels=256 // factor, out_channels=256 // factor, mid_channels=None,
                                          kernel_size=3)

        self.up1 = Complex_Up(in_channels=256, out_channels=128 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.up2 = Complex_Up(in_channels=128, out_channels=64 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.up3 = Complex_Up(in_channels=64, out_channels=32 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.up4 = Complex_Up(in_channels=32, out_channels=16, bilinear=self.bilinear, Conv2d_kernel_size=3,
                              ConvTrans_kernel_size=2)
        self.outc1 = Complex_OutConv(in_channels=16, out_channels=n_out_channels, kernel_size=1)



        ############### Classifier

        # self.flatten = Complex_Flatten()
        # self.inc2 = Complex_Linear(in_channels=4608, out_channels=2000, activation="relu")
        # self.hidden1 = Complex_Linear(in_channels=2000, out_channels=1000, activation="relu")
        # self.hidden2 = Complex_Linear(in_channels=1000, out_channels=500, activation="relu")
        # self.hidden3 = Complex_Linear(in_channels=500, out_channels=250, activation="relu")
        # self.hidden4 = Complex_Linear(in_channels=250, out_channels=100, activation="relu")
        # self.outc2 = Complex_Linear(in_channels=100, out_channels=n_classes, activation="sigmoid")
        # self.outc3 = Complex_Softmax()

        self.conv1 = ComplexConv2d(in_channels=128, out_channels=200, kernel_size=3, stride=1, padding='same',
                      dilation=1, groups=1, bias=True)
        self.ReLU1 = ComplexReLU()
        self.pool1 = Complexavg_pool2d(kernel_size=2, stride=None, padding=0,
                              dilation=1, return_indices=False, ceil_mode=False,
                              count_include_pad=True, divisor_override=None)
        self.conv2 = ComplexConv2d(in_channels=200, out_channels=256, kernel_size=3, stride=1, padding='same',
                                     dilation=1, groups=1, bias=True)
        self.ReLU2 = ComplexReLU()
        self.flat1 = Complex_Flatten()
        self.lin1 = Complex_Linear(in_channels=2304, out_channels=2000, activation="relu")
        self.lin2 = Complex_Linear(in_channels=2000, out_channels=1000, activation="relu")
        self.lin3 = Complex_Linear(in_channels=1000, out_channels=500, activation="relu")
        self.lin4 = Complex_Linear(in_channels=500, out_channels=100, activation="relu")
        self.out1 = Complex_Linear(in_channels=100, out_channels=n_classes, activation="None")



    def forward(self, x):

        i1 = self.inc1(x)
        d1 = self.down1(i1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        f = self.feature(d4)
        if self.feature_extractor == True:
            return f
        else:
            u1 = self.up1(f, d3)
            u2 = self.up2(u1, d2)
            u3 = self.up3(u2, d1)
            u4 = self.up4(u3, i1)
            reconstructed = self.outc1(u4)

            # f = self.flatten(f)
            # c1 = self.inc2(f)
            # c2 = self.hidden1(c1)
            # c3 = self.hidden2(c2)
            # c4 = self.hidden3(c3)
            # c5 = self.hidden4(c4)
            # o1 = self.outc2(c5)
            # classified = self.outc3(o1)

            c1 = self.ReLU1(self.conv1(f))
            c2 = self.pool1(c1)
            c3 = self.ReLU2(self.conv2(c2))
            c4 = self.flat1(c3)
            c5 = self.lin1(c4)
            c6 = self.lin2(c5)
            c7 = self.lin3(c6)
            c8 = self.lin4(c7)
            classified = abs(self.out1(c8))

            return reconstructed, classified