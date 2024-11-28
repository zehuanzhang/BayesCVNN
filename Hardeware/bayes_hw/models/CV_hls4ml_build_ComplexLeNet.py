#!/usr/bin/env python3
import sys
sys.path.append('../../converter/keras/')
# sys.path.append('../Hardware_Artifact/')

import keras 
from converter.keras.MCDropout import MCDropout, BayesianDropout
from converter.keras.BernoulliDropout import BernoulliDropout
# from converter.keras.Slicing import Fhalf
from converter.keras.ID import ID

from keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import argparse 
import os
from tensorflow.keras.models import load_model
import numpy as np
import math
from model_utils import *

from qkeras.qconvolutional import QComplexConv2D, QComplexDense, QComplexAveragePooling2D, QComplexActivation, ComplexSoftmaxActivation, QComplexBatchNormalization2d
from keras.layers import AveragePooling2D, Concatenate

def convert_build(args):
    co = {"BayesianDropout": BayesianDropout,
          "MCDropout": MCDropout,
          "BernoulliDropout": BernoulliDropout,
          "ID": ID
          # "ComplexBayesianDropout": ComplexBayesianDropout,
          # "QComplexConv2D": QComplexConv2D,
          # "QComplexDense": QComplexDense,
          # # "QComplexAveragePooling2D": QComplexAveragePooling2D,
          # "QComplexActivation": QComplexActivation,
          # "ComplexSoftmaxActivation": ComplexSoftmaxActivation,
          # "QComplexBatchNormalization2d": QComplexBatchNormalization2d,
          # "ComplexAveragePooling2D": ComplexAveragePooling2D
          }
    _add_supported_quantized_objects(co)

    # args.load_model = './saved_model_rmultiply/model_multiply'
    # model = load_model(args.load_model + '.h5', custom_objects=co)
    # args.output_dir = './hls_model_rmultiply'

    args.load_model = './saved_model_ComplexLeNet/model_ComplexLeNet'
    model = load_model(args.load_model + '.h5', custom_objects=co)
    args.output_dir = './hls_model_ComplexLeNet'
    # sys.exit()

    model.summary()
    # sys.exit()

    import hls4ml
    #import plotting

    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Activation'])
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(rounding_mode='AP_RND')
    hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(saturation_mode='AP_SAT')

    #First, the baseline model
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

    # Set the precision and reuse factor for the full model

    hls_config['Model']['ReuseFactor'] = 1
    hls_config['Model']['Strategy'] = 'Latency'

    hls_config['Model']['BramFactor'] = 50000
    hls_config['Model']['MergeFactor'] = 1

    cfg = hls4ml.converters.create_config(backend='Vivado')
    cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
    cfg['HLSConfig']  = hls_config
    cfg['KerasModel'] = model
    cfg['OutputDir']  = args.output_dir + '/'
    cfg['XilinxPart'] = 'xcku115-flvb2104-2-i'
    cfg['Part'] = 'xcku115-flvb2104-2-i'
    cfg['Bayes'] = True
    cfg['ClockPeriod'] = 5.5

    hls_model = hls4ml.converters.keras_to_hls(cfg)
    # sys.exit()
    hls_model.compile()
    # sys.exit()
    hls_model.build(csim=False, synth=True, vsynth=True, export=True)


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_model", default=f'./saved_model/CVmodel', type=str, help="Name of load model")
    parser.add_argument("--quant_ibit", default=0, type=int, help="The integer bits of quant")
    parser.add_argument("--num_bayes_layer", default=1, type=int, help="The number of Bayesian Layer")
    parser.add_argument("--output_dir", default='temp', type=str, help="Output directory")
    parser.add_argument("--strategy", default='resource', type=str, help="Stategy for implmenetation, latency or resource")
    parser.add_argument("--num_bins", default=10, type=int, help="The number of bins while calculating ECE")
    parser.add_argument("--num_mc_samples", default=1, type=int, help="The number of MC samples")
    parser.add_argument("--dropout_rate", default=0.25, type=float, help="The dropout rate")
    parser.add_argument("--num_masks", default=4, type=int, help="The number of masks")
    parser.add_argument("--scale", default=4, type=float, help="The scale")
    parser.add_argument("--mem_limit", default=4096, type=int, help="Mem limit for implementation")
    parser.add_argument("--dropout_type", default="mc", type=str, choices=["mc"], help="Dropout type, Monte-Carlo Dropout (mc)")

    args = parser.parse_args()

    convert_build(args)
