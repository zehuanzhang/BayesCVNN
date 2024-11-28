import sys
sys.path.append('/mnt/ccnas2/bdp/zz3822/BayesCVNN_hls4ml/Hardware_Artifact/converter/keras/')
sys.path.append('/mnt/ccnas2/bdp/zz3822/BayesCVNN_hls4ml/Hardware_Artifact/')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Subtract, Reshape, Lambda
from qkeras import QConv2D, QDense, QActivation, quantized_bits, quantized_relu
from keras.layers import AveragePooling2D, MaxPooling2D
from qkeras.qnormalization import QBatchNormalization

import numpy as np
from converter.keras.MCDropout import BayesianDropout, MCDropout
from converter.keras.BernoulliDropout import BernoulliDropout

import tensorflow as tf
import hls4ml

# Define the complex activation function
def complex_activation(input_real, input_imag, activation=quantized_relu(6)):

    output_real = QActivation(activation=activation)(input_real)
    output_imag = QActivation(activation=activation)(input_imag)

    return output_real, output_imag

def complex_avgpool2d(input_real, input_imag, pool_size=2):

    output_real = AveragePooling2D(pool_size=pool_size)(input_real)
    output_imag = AveragePooling2D(pool_size=pool_size)(input_imag)
    return output_real, output_imag

def complex_maxpool2d(input_real, input_imag, pool_size=2):

    output_real = MaxPooling2D(pool_size=pool_size)(input_real)
    output_imag = MaxPooling2D(pool_size=pool_size)(input_imag)
    return output_real, output_imag

def complex_batchnorm(input_real, input_imag):

    output_real = QBatchNormalization()(input_real)
    output_imag = QBatchNormalization()(input_imag)

    return output_real, output_imag

def complex_bayesiandropout(input_real, input_imag, drop_rate):
    """
    Applies Bayesian Dropout to the real and imaginary parts of the input based on the 'real_imag' index.
    """

    output_real = BayesianDropout(drop_rate=drop_rate)(input_real)
    output_imag = BayesianDropout(drop_rate=drop_rate)(input_imag)

    return output_real, output_imag

def complex_bernoullidropout(input_real, input_imag, drop_rate, sw_real, sw_imag):
    """
    Applies Bernoulli Dropout to the real and imaginary parts of the input based on the 'real_imag' index.
    """

    output_real = BernoulliDropout(drop_rate=drop_rate)(input_real) if sw_real else input_real
    output_imag = BernoulliDropout(drop_rate=drop_rate)(input_imag) if sw_imag else input_imag

    return output_real, output_imag

# Define the complex convolution function
def complex_conv2d(input_real, input_imag, filters, kernel_size=3, padding='same', kernel_quantizer=quantized_bits(6, 0, alpha=1), bias_quantizer=quantized_bits(6, 0, alpha=1)):
    # Define Conv_real and Conv_imag layers
    ## Real part convolutions


    conv_real_real = QConv2D(filters=filters,
                             kernel_size=kernel_size,
                             padding=padding,
                             kernel_quantizer=kernel_quantizer,
                             bias_quantizer=bias_quantizer)(input_real)

    conv_real_imag = QConv2D(filters=filters,
                             kernel_size=kernel_size,
                             padding=padding,
                             kernel_quantizer=kernel_quantizer,
                             bias_quantizer=bias_quantizer)(input_imag)

    conv_imag_real = QConv2D(filters=filters,
                             kernel_size=kernel_size,
                             padding=padding,
                             kernel_quantizer=kernel_quantizer,
                             bias_quantizer=bias_quantizer)(input_real)

    conv_imag_imag = QConv2D(filters=filters,
                             kernel_size=kernel_size,
                             padding=padding,
                             kernel_quantizer=kernel_quantizer,
                             bias_quantizer=bias_quantizer)(input_imag)

    # Combine real and imaginary outputs
    output_real = Subtract()([conv_real_real, conv_imag_imag])  # Real part of the output
    output_imag = Add()([conv_real_imag, conv_imag_real])  # Imaginary part of the output

    return output_real, output_imag

def complex_dense(input_real, input_imag, units, kernel_quantizer=quantized_bits(6, 0, alpha=1), bias_quantizer=quantized_bits(6, 0, alpha=1)):
    # Define Conv_real and Conv_imag layers

    dense_real_real = QDense(units=units,
                             kernel_quantizer=kernel_quantizer,
                             bias_quantizer=bias_quantizer)(input_real)

    dense_real_imag = QDense(units=units,
                             kernel_quantizer=kernel_quantizer,
                             bias_quantizer=bias_quantizer)(input_imag)

    dense_imag_real = QDense(units=units,
                             kernel_quantizer=kernel_quantizer,
                             bias_quantizer=bias_quantizer)(input_real)

    dense_imag_imag = QDense(units=units,
                             kernel_quantizer=kernel_quantizer,
                             bias_quantizer=bias_quantizer)(input_imag)

    # Combine real and imaginary outputs
    output_real = Subtract()([dense_real_real, dense_imag_imag])
    output_imag = Add()([dense_real_imag, dense_imag_real])

    return output_real, output_imag
