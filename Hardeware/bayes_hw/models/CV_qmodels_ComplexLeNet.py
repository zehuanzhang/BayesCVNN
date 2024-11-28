import sys
sys.path.append('../../converter/keras/')
sys.path.append('../')
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import *
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv2DBatchnorm
from qkeras.qpooling import QAveragePooling2D
from qkeras.qnormalization import QBatchNormalization
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
from tensorflow.keras.utils import to_categorical
from qkeras import *
from tensorflow.keras.optimizers import Adam, SGD
from converter.keras.MCDropout import BayesianDropout
from converter.keras.Masksembles import Masksembles
from keras.regularizers import l2, l1
from keras import layers
import os
import argparse
import numpy as np

from re import X
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, MaxPool2D, GlobalAveragePooling2D, Reshape, Concatenate, \
    Add, Subtract
import keras.backend as K
import math
from model_utils import Insert_Bayesian_Layer, Bayesian_Layer
from converter.keras.BernoulliDropout import BernoulliDropout

import copy

from converter.keras_complex.CV_qmodels_rfunc import (complex_activation, complex_avgpool2d, complex_bayesiandropout,
                                                      complex_bernoullidropout, complex_conv2d, complex_dense,
                                                      complex_batchnorm, complex_maxpool2d)


# Define the ComNet model
def BayesComplexLeNet(input_shape=(28, 28, 1)):
    # Define inputs for real and imaginary parts
    input_real = Input(shape=input_shape)
    input_imag = Input(shape=input_shape)

    # Use the complex convolution block
    output_real, output_imag = complex_conv2d(input_real, input_imag, filters=6, kernel_size=5)
    output_real, output_imag = complex_batchnorm(output_real, output_imag)
    output_real, output_imag = complex_activation(output_real, output_imag)
    output_real, output_imag = complex_maxpool2d(output_real, output_imag)
    output_real, output_imag = complex_bernoullidropout(output_real, output_imag, drop_rate=0.25, sw_real=False, sw_imag=True)

    output_real, output_imag = complex_conv2d(output_real, output_imag, filters=16, kernel_size=5)
    output_real, output_imag = complex_batchnorm(output_real, output_imag)
    output_real, output_imag = complex_activation(output_real, output_imag)
    output_real, output_imag = complex_maxpool2d(output_real, output_imag)


    output_real = Flatten()(output_real)
    output_imag = Flatten()(output_imag)

    output_real, output_imag = complex_dense(output_real, output_imag, units=84)
    output_real, output_imag = complex_activation(output_real, output_imag)
    output_real, output_imag = complex_dense(output_real, output_imag, units=10)

    # # real_squared = Multiply()([output_real, output_real])
    # # imag_squared = Multiply()([output_imag, output_imag])
    # # squared_modulus = Add()([real_squared, imag_squared])
    # # out_softmax = Activation(activation='softmax', name='softmax')(squared_modulus)
    # out_softmax = Add()([output_real, output_imag])


    model = Model(inputs=[input_real, input_imag], outputs=[output_real, output_imag])#out_softmax)

    return model

# Define the ComNet model
def BayesComplexLeNet_small(input_shape=(28, 28, 1)):
    # Define inputs for real and imaginary parts
    input_real = Input(shape=input_shape)
    input_imag = Input(shape=input_shape)

    # Use the complex convolution block
    output_real, output_imag = complex_conv2d(input_real, input_imag, filters=3, kernel_size=5)
    output_real, output_imag = complex_maxpool2d(output_real, output_imag)
    output_real, output_imag = complex_bernoullidropout(output_real, output_imag, drop_rate=0.25, sw_real=True, sw_imag=True)


    output_real = Flatten()(output_real)
    output_imag = Flatten()(output_imag)

    output_real, output_imag = complex_dense(output_real, output_imag, units=10)



    model = Model(inputs=[input_real, input_imag], outputs=[output_real, output_imag])#out_softmax)

    return model

# Define the ComNet model
def ComplexLeNet(input_shape=(28, 28, 1)):
    # Define inputs for real and imaginary parts
    input_real = Input(shape=input_shape)
    input_imag = Input(shape=input_shape)

    # Use the complex convolution block
    output_real, output_imag = complex_conv2d(input_real, input_imag, filters=6, kernel_size=5)
    output_real, output_imag = complex_batchnorm(output_real, output_imag)
    output_real, output_imag = complex_activation(output_real, output_imag)
    output_real, output_imag = complex_maxpool2d(output_real, output_imag)

    output_real, output_imag = complex_conv2d(output_real, output_imag, filters=16, kernel_size=5)
    output_real, output_imag = complex_batchnorm(output_real, output_imag)
    output_real, output_imag = complex_activation(output_real, output_imag)
    output_real, output_imag = complex_maxpool2d(output_real, output_imag)


    output_real = Flatten()(output_real)
    output_imag = Flatten()(output_imag)

    output_real, output_imag = complex_dense(output_real, output_imag, units=84)
    output_real, output_imag = complex_activation(output_real, output_imag)
    output_real, output_imag = complex_dense(output_real, output_imag, units=10)

    # # real_squared = Multiply()([output_real, output_real])
    # # imag_squared = Multiply()([output_imag, output_imag])
    # # squared_modulus = Add()([real_squared, imag_squared])
    # # out_softmax = Activation(activation='softmax', name='softmax')(squared_modulus)
    # out_softmax = Add()([output_real, output_imag])


    model = Model(inputs=[input_real, input_imag], outputs=[output_real, output_imag])#out_softmax)

    return model




# Define the ComNet model
def RealLeNet(input_shape=(28, 28, 1)):
    # Define inputs for real and imaginary parts
    input = Input(shape=input_shape)

    # Use the complex convolution block
    output = QConv2D(filters=6, kernel_size=5, padding='same', kernel_quantizer=quantized_bits(6, 0, alpha=1), bias_quantizer=quantized_bits(6, 0, alpha=1))(input)
    output = QBatchNormalization()(output)
    output = QActivation(activation=quantized_relu(6))(output)
    output = AveragePooling2D(pool_size=2)(output)
    # output = BernoulliDropout(drop_rate=0.25)(output)

    output = QConv2D(filters=16, kernel_size=5, padding='same', kernel_quantizer=quantized_bits(6, 0, alpha=1), bias_quantizer=quantized_bits(6, 0, alpha=1))(output)
    output = QBatchNormalization()(output)
    output = QActivation(activation=quantized_relu(6))(output)
    output = AveragePooling2D(pool_size=2)(output)
    # output = BernoulliDropout(drop_rate=0.25)(output)

    output = Flatten()(output)

    output = QDense(units=84, kernel_quantizer=quantized_bits(6, 0, alpha=1), bias_quantizer=quantized_bits(6, 0, alpha=1))(output)
    # output = BayesianDropout(drop_rate=0.25)(output)
    output = QDense(units=10, kernel_quantizer=quantized_bits(6, 0, alpha=1), bias_quantizer=quantized_bits(6, 0, alpha=1))(output)

    # # real_squared = Multiply()([output_real, output_real])
    # # imag_squared = Multiply()([output_imag, output_imag])
    # # squared_modulus = Add()([real_squared, imag_squared])
    # # out_softmax = Activation(activation='softmax', name='softmax')(squared_modulus)
    # out_softmax = Add()([output_real, output_imag])


    model = Model(inputs=[input], outputs=[output])#out_softmax)

    return model






# Instantiate the model with input shape (100, 100, 2)
# model = BayesComplexLeNet()
# model = BayesComplexLeNet_small()
model = ComplexLeNet()
# model = RealLeNet()
# Print the model summary
model.summary()
model.compile(optimizer=SGD(lr=0.0001), loss=['categorical_crossentropy', 'categorical_crossentropy'],
              metrics=['accuracy'])
# model.save(f'./saved_model_ComplexLeNet/model_ComplexLeNeth5')
# model.save(f'./saved_model_ComplexLeNet/model_ComplexLeNet_dropout.h5')
# model.save(f'./saved_model_ComplexLeNet/model_ComplexLeNet_real.h5')
# model.save(f'./saved_model_ComplexLeNet/model_ComplexLeNet_1layer_I.h5')
model.save(f'./saved_model_ComplexLeNet/model_ComplexLeNet.h5')

