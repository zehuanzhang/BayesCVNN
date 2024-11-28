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
# Define the complex dense function
def complex_dense(input_real, input_imag, units, kernel_quantizer, bias_quantizer):
    # Define Conv_real and Conv_imag layers
    Dense_real_1 = QDense(units=units,
                        kernel_quantizer=kernel_quantizer, bias_quantizer=bias_quantizer)
    output_real_real = Dense_real_1(input_real)

    Dense_real_2 = QDense(units=units,
                          kernel_quantizer=kernel_quantizer, bias_quantizer=bias_quantizer)
    _ = Dense_real_2(input_imag)
    Dense_real_2.set_weights(Dense_real_1.get_weights())
    output_real_imag = Dense_real_2(input_imag)

    Dense_imag_1 = QDense(units=units,
                          kernel_quantizer=kernel_quantizer, bias_quantizer=bias_quantizer)
    output_imag_real = Dense_imag_1(input_real)

    Dense_imag_2 = QDense(units=units,
                          kernel_quantizer=kernel_quantizer, bias_quantizer=bias_quantizer)
    _ = Dense_imag_2(input_imag)
    Dense_imag_2.set_weights(Dense_imag_1.get_weights())
    output_imag_imag = Dense_imag_2(input_imag)

    # Combine real and imaginary outputs
    output_real = Subtract()([output_real_real, output_imag_real])
    output_imag = Add()([output_real_imag, output_imag_imag])

    return output_real, output_imag


################################################################################################################################################


# Define the ComNet model
def ComNet(input_shape=(28, 28, 1)):
    # Define inputs for real and imaginary parts
    # input = Input(shape=input_shape)
    input_real = Input(shape=input_shape)
    input_imag = Input(shape=input_shape)

    # Use the complex convolution block
    output_real, output_imag = complex_dense(input_real, input_imag, units=4,
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
model.model.save(f'./saved_model_rdense/CVmodel_rdense.h5')

