"""
Based on "R. M.Asiyabi, M. Datcu, A. Anghel, H. Nies, "Complex-Valued End-to-end Deep
 Network with Coherency Preservation for Complex-Valued SAR Data Reconstruction and
  Classification" IEEE Transactions on Geoscience and Remote Sensing (2023)"
"""

import argparse
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import imageio
import glob

sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torchvision
import torchvision.transforms as transforms

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from ComplexValuedAutoencoderMain_Torch import end_to_end_Net_classification_small_Bayesian, \
    end_to_end_Net_classification
from Src.ComplexValuedAutoencoder_Class_Torch import Complex2foldloss, Complex2foldloss_Coh
from Src.Dataset_Preperation_Torch import BasicDataset
from utils import evaluate

import copy
import csv
import itertools

torch.autograd.set_detect_anomaly(True)

dir_checkpoint = 'checkpoints/'


def dataset_func(dataset_name,
                 batch_size,
                 img_scale,
                 save_cp=True
                 ):
    ############ Datasets

    train_dataset = np.load(f"./dataset_random/{dataset_name}/{dataset_name}_train_dataset.npy")
    train_labels = np.load(f"./dataset_random/{dataset_name}/{dataset_name}_train_labels.npy")
    test_dataset = np.load(f"./dataset_random/{dataset_name}/{dataset_name}_test_dataset.npy")
    test_labels = np.load(f"./dataset_random/{dataset_name}/{dataset_name}_test_labels.npy")

    n_train = train_dataset.shape[0]
    n_test = test_dataset.shape[0]

    print("Train Labels and their counts:", np.unique(train_labels, return_counts=True))
    print("Test Labels and their counts:", np.unique(test_labels, return_counts=True))

    datadict_train = BasicDataset(imgs=train_dataset, labels=train_labels, scale=img_scale, normal=False)
    train_loader = DataLoader(datadict_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    datadict_test = BasicDataset(imgs=test_dataset, labels=test_labels, scale=img_scale, normal=False)
    test_loader = DataLoader(datadict_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logging.info(f'''Starting training:
        Batch size:      {batch_size}
        Training size:   {n_train}
        Validation size: {n_test}
        Checkpoints:     {save_cp}
        Images scaling:  {img_scale}
    ''')

    return n_train, train_loader, n_test, test_loader


def test_net(net,
             device,
             test_loader,
             samples_number=1
             ):

    ##test
    net.eval()

    with torch.no_grad():
        outputs = []
        labels = []
        for batch in test_loader:
            imgs = batch["image"]
            label = batch["label"] - 1

            imgs = imgs.to(device=device, dtype=torch.complex64)
            label = label.squeeze(dim=1).to(device=device, dtype=torch.long)

            samples = []
            for j in range(samples_number):
                output = net(imgs)
                samples.append(output)
            outputs.append(torch.stack(samples, dim=1).mean(dim=1))
            labels.append(label)

        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)

        error, ece, entropy, loss = evaluate(output=outputs, target=labels)  # outputs:torch.Size([10000, 10])targets:torch.Size([10000])
        print('acc, ece, loss:', 100 - error, ece, loss)

    return 100 - error, ece


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=30,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-d', '--dataset_name', type=str, default='Houston',  # "Sao_Paulo"
                        help='Name of places')
    parser.add_argument('-g', '--group_number', type=int, default=0,
                        help='Number of groups')
    parser.add_argument('-n', '--samples_number', type=int, default=3,
                        help='Number of samples')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    layer_settings_total = list(itertools.product([0, 1, 2, 3], repeat=3))
    layer_settings_total = [item for item in layer_settings_total if item != (0, 0, 0)]

    # n=9
    # k, m = divmod(len(layer_settings_total), n)
    # groups = [layer_settings_total[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    # layer_settings = groups[args.group_number]
    #
    # #python Train_class7_Bayesian.py --dataset_name 'Chicago'
    # #echo 0 1 2 3 4 5 6 7 8 | xargs -n 1 -P 9 -I {} python Train_class7_Bayesian.py --dataset name 'Chicago' --group_number {}
    # #Chicago Houston Sao_Paulo

    with open(f'{args.dataset_name}_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.dataset_name, 'samples_number', 'layer_setting', 'accuracy', 'ece'])


    for layer_setting in layer_settings_total:
        model_path = f'./{args.dataset_name}_trained_models_class7_test_Bayesian/{layer_setting}/{args.dataset_name}_model_epoch30.pth'
        net = torch.load(model_path, map_location=device)

        n_train, train_loader, n_test, test_loader = dataset_func(dataset_name=args.dataset_name, batch_size=args.batchsize,
                                                                  img_scale=args.scale)
        print('args.dataset_name:', args.dataset_name)
        print('args.batchsize:', args.batchsize)
        print('args.val / 100:', args.val / 100)
        print('args.scale:', args.scale)
        print('args.epochs:', args.epochs)

        accuracy, ece = test_net(net=net, device=device, test_loader=test_loader, samples_number=args.samples_number)

        with open(f'{args.dataset_name}_results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([args.dataset_name, args.samples_number, layer_setting, accuracy, ece])



    # torch.save(net.state_dict(), 'path to save the trained model.pth')
