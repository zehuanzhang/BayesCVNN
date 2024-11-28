import sys

import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.quantized import QFunctional
import itertools


class BernoulliDropout(nn.Module):
    def __init__(self, p=0.0, layer_setting=0):
        super(BernoulliDropout, self).__init__()
        self.p = torch.nn.Parameter(torch.ones((1,))*p, requires_grad=False)
        self.setting = (False, True)
        self.layer_setting = self.setting[layer_setting]
        if self.p < 1:
            self.multiplier = torch.nn.Parameter(
                torch.ones((1,))/(1.0 - self.p), requires_grad=False)
        else:
            self.multiplier = torch.nn.Parameter(
                torch.ones((1,))*0.0, requires_grad=False)

        self.mul_mask = torch.nn.quantized.FloatFunctional()
        self.mul_scalar = torch.nn.quantized.FloatFunctional()
        
    def forward(self, x):
        if self.p <= 0.0:
            return x
        if not self.layer_setting:
            return x
        mask_ = None
        if len(x.shape) <= 2:
            if x.is_cuda:
                mask_ = torch.cuda.FloatTensor(x.shape).bernoulli_(1.-self.p)
            else:
                mask_ = torch.FloatTensor(x.shape).bernoulli_(1.-self.p)
        else:
            if x.is_cuda:
                mask_ = torch.cuda.FloatTensor(x.shape[:2]).bernoulli_(
                    1.-self.p)
            else:
                mask_ = torch.FloatTensor(x.shape[:2]).bernoulli_(
                    1.-self.p)
        if isinstance(self.mul_mask, QFunctional):
            scale = self.mul_mask.scale
            zero_point = self.mul_mask.zero_point
            mask_ = torch.quantize_per_tensor(
                mask_, scale, zero_point, dtype=torch.quint8)
        if len(x.shape) > 2:
            mask_ = mask_.view(
                mask_.shape[0], mask_.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        x = self.mul_mask.mul(x, mask_)
        x = self.mul_scalar.mul_scalar(x, self.multiplier.item())
        return x

    def extra_repr(self):
        return 'p={}, quant={}'.format(
            self.p.item(), isinstance(
                self.mul_mask, QFunctional)
        )


class Complex_BernoulliDropout(nn.Module):
    def __init__(self, p=0.0, real_imag = 0):
        super(Complex_BernoulliDropout, self).__init__()
        self.p = torch.nn.Parameter(torch.ones((1,)) * p, requires_grad=False)
        self.setting = [(False, False), (False, True), (True, False), (True, True)]
        self.real_imag = self.setting[real_imag]

        if self.p < 1:
            self.multiplier = torch.nn.Parameter(
                torch.ones((1,)) / (1.0 - self.p), requires_grad=False)
        else:
            self.multiplier = torch.nn.Parameter(
                torch.ones((1,)) * 0.0, requires_grad=False)

        self.mul_mask = torch.nn.quantized.FloatFunctional()
        self.mul_scalar = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.p <= 0.0:
            return x
        mask_ = None
        if len(x.shape) <= 2:
            if x.is_cuda:
                real_mask_ = torch.cuda.FloatTensor(x.shape).bernoulli_(1. - self.p)
                imag_mask_ = torch.cuda.FloatTensor(x.shape).bernoulli_(1. - self.p)
            else:
                real_mask_ = torch.FloatTensor(x.shape).bernoulli_(1. - self.p)
                imag_mask_ = torch.cuda.FloatTensor(x.shape).bernoulli_(1. - self.p)
        else:
            if x.is_cuda:
                real_mask_ = torch.cuda.FloatTensor(x.shape[:2]).bernoulli_(
                    1. - self.p)
                imag_mask_ = torch.cuda.FloatTensor(x.shape[:2]).bernoulli_(
                    1. - self.p)
            else:
                real_mask_ = torch.FloatTensor(x.shape[:2]).bernoulli_(
                    1. - self.p)
                imag_mask_ = torch.cuda.FloatTensor(x.shape[:2]).bernoulli_(
                    1. - self.p)

        if isinstance(self.mul_mask, QFunctional):
            scale = self.mul_mask.scale
            zero_point = self.mul_mask.zero_point
            real_mask_ = torch.quantize_per_tensor(
                real_mask_, scale, zero_point, dtype=torch.quint8)
            imag_mask_ = torch.quantize_per_tensor(
                imag_mask_, scale, zero_point, dtype=torch.quint8)
        if len(x.shape) > 2:
            real_mask_ = real_mask_.view(
                real_mask_.shape[0], real_mask_.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
            imag_mask_ = imag_mask_.view(
                imag_mask_.shape[0], imag_mask_.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])

        if self.real_imag[0]:
            x.real = self.mul_mask.mul(x.real, real_mask_)
            x.real = self.mul_scalar.mul_scalar(x.real, self.multiplier.item())
        if self.real_imag[1]:
            x.imag = self.mul_mask.mul(x.imag, imag_mask_)
            x.imag = self.mul_scalar.mul_scalar(x.imag, self.multiplier.item())
        return x

    def extra_repr(self):
        return 'p={}, quant={}'.format(
            self.p.item(), isinstance(
                self.mul_mask, QFunctional)
        )