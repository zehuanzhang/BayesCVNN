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
# Define the complex activation function
def complex_activation(input_real, input_imag, activation):

    output_real = QActivation(activation=activation)(input_real)
    output_imag = QActivation(activation=activation)(input_imag)

    return output_real, output_imag


################################################################################################################################################


# Define the ComNet model
def ComNet(input_shape=(28, 28, 1)):
    # Define inputs for real and imaginary parts
    # input = Input(shape=input_shape)
    input_real = Input(shape=input_shape)
    input_imag = Input(shape=input_shape)

    # Use the complex convolution block
    output_real, output_imag = complex_activation(input_real, input_imag, activation=quantized_relu(6))

    model = Model(inputs=[input_real, input_imag], outputs=[output_real, output_imag])

    return model

# Instantiate the model with input shape (100, 100, 2)
model = ComNet()
# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model = MCDropout(model, nSamples=3, p=0.25, num=0)
model.model.save(f'./saved_model_ractivation/CVmodel_ractivation.h5')

