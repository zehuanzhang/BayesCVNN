import sys
sys.path.append('/mnt/ccnas2/bdp/zz3822/BayesCVNN_hls4ml/Hardware_Artifact/converter/keras/')
sys.path.append('/mnt/ccnas2/bdp/zz3822/BayesCVNN_hls4ml/Hardware_Artifact/')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Subtract, Reshape, Lambda
from qkeras import QConv2D, QDense, QActivation, quantized_bits, quantized_relu
from keras.layers import AveragePooling2D

import numpy as np
from converter.keras.BernoulliDropout import BernoulliDropout
from converter.keras.MCDropout import BayesianDropout, MCDropout

import tensorflow as tf
import hls4ml
################################################################################################################################################
# Define the ComNet model with complex BernoulliDropout

def complex_bernoullidropout(input_real, input_imag, drop_rate):
    """
    Applies Bayesian Dropout to the real and imaginary parts of the input based on the 'real_imag' index.
    """

    output_real = BernoulliDropout(drop_rate=drop_rate)(input_real)
    output_imag = BernoulliDropout(drop_rate=drop_rate)(input_imag)

    return output_real, output_imag
################################################################################################################################################

def ComNet(input_shape=(28, 28, 1)):
    # Define inputs for real and imaginary parts
    # input = Input(shape=input_shape)
    input_real = Input(shape=input_shape)
    input_imag = Input(shape=input_shape)

    # Use the complex convolution block
    output_real, output_imag = complex_bernoullidropout(input_real, input_imag, drop_rate=0.25)

    model = Model(inputs=[input_real, input_imag], outputs=[output_real, output_imag])

    return model

# Instantiate the model with input shape (28, 28, 10)
model = ComNet()
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model = MCDropout(model, nSamples=3, p=0.25, num=0)
model.model.save(f'./saved_model_rbernoullidropout/CVmodel_rbernoullidropout.h5')





# # Define a function to check for zero channels
# def check_zero_channels(output):
#     # Count how many channels have all zeros
#     print(output.shape)
#     zero_channels = np.sum(np.all(output == 0, axis=(1, 2)))  # Count channels where all values are zero
#     print(f"Number of zero channels: {zero_channels}")
#
# # Create some dummy data to pass through the model
# input_data = np.random.randn(1, 28, 28, 1).astype(np.float32)  # A batch of 1 with shape (28, 28, 10)
#
# # Use the model to predict and get the output
# output_data = model(input_data, training=True)  # Set training=True to apply dropout
#
# # Check how many channels are zero in the output
# output_data = output_data.numpy()  # Convert TensorFlow tensor to numpy array
# check_zero_channels(output_data)

