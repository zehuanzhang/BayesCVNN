import keras
from keras import Model
import numpy as np
import tensorflow as tf
from . import nn2bnn
from nn2bnn import _convert_model, strategy_fn, HlsLayer

import tensorflow as tf
from keras import layers




# Keras implementation of a custom layer
class ID(keras.layers.Layer):
    '''Keras implementation of a hypothetical custom layer'''

    def __init__(self, **kwargs):
        super(ID, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return super(ID, self).get_config()



# # hls4ml layer implementation
# class HFhalf(HlsLayer):
#     _expected_attributes = [
#         Attribute('n_in')
#     ]
#
#     def initialize(self):
#         inp = self.get_input_variable()
#         shape = inp.shape
#         dims = inp.dim_names
#         self.add_output_variable(shape, dims)
#         self.set_attr('n_in', self.get_input_variable().size())
        

