import sys
sys.path.append('/mnt/ccnas2/bdp/zz3822/BayesCVNN_hls4ml/Hardware_Artifact/converter/keras/')
sys.path.append('/mnt/ccnas2/bdp/zz3822/BayesCVNN_hls4ml/Hardware_Artifact/')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Subtract, Reshape, Lambda
from qkeras import QConv2D, QDense, QActivation, quantized_bits, quantized_relu
from keras.layers import AveragePooling2D

import numpy as np
from converter.keras.MCDropout import BayesianDropout, MCDropout

import tensorflow as tf
import hls4ml
################################################################################################################################################

# Define the complex convolution function
def complex_conv2d(input_real, input_imag, filters, kernel_size, padding, kernel_quantizer, bias_quantizer):
    # Define Conv_real and Conv_imag layers
    ## Real part convolutions
    Conv_real_1 = QConv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                          kernel_quantizer=kernel_quantizer, bias_quantizer=bias_quantizer)
    output_real_real = Conv_real_1(input_real)  # Conv_real_1 processes the real input

    Conv_real_2 = QConv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                          kernel_quantizer=kernel_quantizer, bias_quantizer=bias_quantizer)
    _ = Conv_real_2(input_imag)  # Conv_real_2 processes the imaginary input
    Conv_real_2.set_weights(Conv_real_1.get_weights())  # Copy Conv_real_1's weights to Conv_real_2
    output_real_imag = Conv_real_2(input_imag)  # Now apply Conv_real_2 with the copied weights

    ## Imaginary part convolutions
    Conv_imag_1 = QConv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                          kernel_quantizer=kernel_quantizer, bias_quantizer=bias_quantizer)
    output_imag_real = Conv_imag_1(input_real)  # Conv_imag_1 processes the imaginary input

    Conv_imag_2 = QConv2D(filters=filters, kernel_size=kernel_size, padding=padding,
                          kernel_quantizer=kernel_quantizer, bias_quantizer=bias_quantizer)
    _ = Conv_imag_2(input_imag)  # Conv_imag_2 processes the real input
    Conv_imag_2.set_weights(Conv_imag_1.get_weights())  # Copy Conv_imag_1's weights to Conv_imag_2
    output_imag_imag = Conv_imag_2(input_imag)  # Now apply Conv_imag_2 with the copied weights

    # Combine real and imaginary outputs
    output_real = Subtract()([output_real_real, output_imag_real])  # Real part of the output
    output_imag = Add()([output_real_imag, output_imag_imag])  # Imaginary part of the output

    return output_real, output_imag


################################################################################################################################################


# Define the ComNet model
def ComNet(input_shape=(28, 28, 1)):
    # Define inputs for real and imaginary parts
    # input = Input(shape=input_shape)
    input_real = Input(shape=input_shape)
    input_imag = Input(shape=input_shape)

    # Use the complex convolution block
    output_real, output_imag = complex_conv2d(input_real, input_imag, filters=4, kernel_size=3, padding='same',
                                              kernel_quantizer=quantized_bits(6, 0, alpha=1),
                                              bias_quantizer=quantized_bits(6, 0, alpha=1))

    model = Model(inputs=[input_real, input_imag], outputs=[output_real, output_imag])

    return model

# Instantiate the model with input shape (100, 100, 2)
model = ComNet()
# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model = MCDropout(model, nSamples=3, p=0.25, num=0)
model.model.save(f'./saved_model_rconv/CVmodel_rconv.h5')

