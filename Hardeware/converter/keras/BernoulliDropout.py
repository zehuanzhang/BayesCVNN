import keras
from keras import Model
import numpy as np
import tensorflow as tf
from nn2bnn import _convert_model, strategy_fn, HlsLayer

import tensorflow as tf
from keras import layers


class BernoulliDropout(tf.keras.layers.Layer):
    r"""
        Applies Dropout to the input.
    """
    def __init__(self, drop_rate=0.5, seed=None, **kwargs):
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(drop_rate))
        super(BernoulliDropout, self).__init__(**kwargs)
        self.drop_rate = drop_rate
        # if seed set to None, random output will be given
        self.seed = seed

    def get_config(self):
        # default seed is 0
        seed = 0 if self.seed is None else self.seed
        config = {
            'drop_rate': self.drop_rate,
            'seed': seed
        }
        base_config = super(BernoulliDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input):
        assert len(input.shape) == 4, \
            "Expected input with 4 dimensions (bsize, height, width, channels)"
        print('Bernoulli:', input.shape)

        mask = tf.cast(tf.greater(tf.random.uniform(shape=[input.shape[-1]], dtype=tf.float32), self.drop_rate), tf.float32)
        mask = tf.reshape(mask, [1, 1, 1, -1])  # Shape: (1, 1, 1, channels)
        mask = tf.broadcast_to(mask, tf.shape(input))  # Broadcast to match the input shape

        # Step 4: Apply the mask
        out = input * mask

        return out


# class BernoulliDropoutModel(HlsLayer):
#     r"""
#      This class uses Morte Carlo Dropout to convert a traditional neural network to a Bayesian neural network.
#      The mathematical proof is in this paper: https://arxiv.org/pdf/1506.02142.pdf. Note that the model to be
#      converted should be contructed using either Sequential or Functional APIs.
#     """
#
#     def __init__(self, model, nSamples=10, p=0.5, num=0,
#                  strategy='default', seed=None, input=None, **kwargs):
#         super().__init__()
#         self.original_model = model
#         supported_layers = strategy_fn[strategy](model, BernoulliDropout, **kwargs)
#         self.model = _convert_model(model, 'BernoulliDropout', supported_layers, p, seed, input) if num > 0 else model
#         self.nSamples = nSamples
#         self.p = p
#         self.seed = seed
#
#     def call(self, input, training=True):
#         if training:
#             return self.model(input, training=True)
#         else:
#             prediction = self.model(input, training=False)
#             if isinstance(prediction, list):
#                 prediction = [prediction[i] for i in range(len(prediction))]
#             else:
#                 pred_shape = prediction.shape
#                 if len(pred_shape) == 2: return prediction  # No MC samples
#                 prediction = [prediction[i] for i in range(pred_shape[0])]
#             return sum(prediction) / len(prediction)
#
#     def get_config(self):
#         config = super(BernoulliDropoutModel, self).get_config()
#         config["model"] = self.model
#         config["nSamples"] = self.nSamples
#         config["p"] = self.p
#         config["seed"] = self.seed
#         return config