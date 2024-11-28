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

from ComplexValuedAutoencoderMain_Torch import end_to_end_Net_classification_small_Bayesian, end_to_end_Net_classification
from Src.ComplexValuedAutoencoder_Class_Torch import Complex2foldloss, Complex2foldloss_Coh
from Src.Dataset_Preperation_Torch import BasicDataset

import copy
import csv
import itertools


torch.autograd.set_detect_anomaly(True)

dir_checkpoint = 'checkpoints/'


def dataset_func(  dataset_name,
              batch_size,
              val_percent,
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

def train_net(net,
              device,
              dataset_name,
              n_train,
              train_loader,
              test_loader,
              epochs,
              batch_size,
              lr,
              val_percent,
              img_scale,
              layer_setting,
              save_cp=True
              ):

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        # Variables to track accuracy
        total_correct = 0
        total_samples = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                imgs = batch["image"]
                label = batch["label"].squeeze(dim=1).long()-1

                # ######### 3-class label classification
                # for l in range(len(label)):
                #     if label[l] == 0 or label[l] == 1:
                #         label[l] = 0
                #     elif label[l] == 2 or label[l] == 3 or label[l] == 4 or label[l] == 5:
                #         label[l] = 1
                #     elif label[l] == 6:
                #         label[l] = 2



                assert imgs.shape[1] == net.n_in_channels, \
                    f'Network has been defined with {net.n_in_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # imgs = imgs.to(device=device, dtype=torch.float32)
                imgs = imgs.to(device=device, dtype=torch.complex64)
                label = label.to(device=device, dtype=torch.long)

                classified_pred = net(imgs)# (10,2,100,100) (10,7)


                loss = criterion(classified_pred, label)
                epoch_loss += loss.item()

                # writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(classified_pred, 1)
                correct = (predicted == label).sum().item()
                total = label.size(0)
                total_correct += correct
                total_samples += total

                pbar.update(imgs.shape[0])

        # Calculate and print the accuracy for the epoch
        epoch_accuracy = total_correct / total_samples
        # print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {total_correct}/{total_samples}, {epoch_accuracy:.4f}')
        os.makedirs(f'./{dataset_name}_trained_models_class7_test_Bayesian/{layer_setting}', exist_ok=True)
        if (epoch + 1) % 10 == 0:
            torch.save(net, f'./{dataset_name}_trained_models_class7_test_Bayesian/{layer_setting}/{dataset_name}_model_epoch{epoch + 1}.pth')
        with open(f'./{dataset_name}_trained_models_class7_test_Bayesian/{layer_setting}/{dataset_name}_epoch_losses.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, epoch_loss, epoch_loss / len(train_loader)])

        ##test
        net.eval()

        outp_classified = []
        label_append = []

        with torch.no_grad():
            #
            #
            # Variables to track accuracy
            total_correct = 0
            total_samples = 0
            for batch in test_loader:
                imgs = batch["image"]
                label = batch["label"] - 1
                label_append.append(label.cpu().detach().numpy())

                imgs = imgs.to(device=device, dtype=torch.complex64)
                label = label.squeeze(dim=1).to(device=device, dtype=torch.long)

                # ######### 3-class label classification
                # for l in range(len(label)):
                #     if label[l] == 0 or label[l] == 1:
                #         label[l] = 0
                #     elif label[l] == 2 or label[l] == 3 or label[l] == 4 or label[l] == 5:
                #         label[l] = 1
                #     elif label[l] == 6:
                #         label[l] = 2

                temp_classified = net(imgs)
                # temp_rect, temp_classified = net(imgs)
                # outp_rect.append(temp_rect.cpu().detach().numpy())
                outp_classified.append(temp_classified.cpu().detach().numpy())  # temp_classified: (10, 3)
                # inp.append(i.cpu().detach().numpy())

                # try:
                #     outp_rect = torch.cat((outp_rect, temp_rect))
                #     # outp_classified = torch.cat((outp_classified, temp_classified))
                #     inp = torch.cat((inp, i))
                # except:
                #     outp_rect = temp_rect
                #     # outp_classified = temp_classified
                #     inp = i

                #
                #
                # Calculate accuracy
                _, predicted = torch.max(temp_classified, 1)
                correct = (predicted == label).sum().item()
                total = label.size(0)
                total_correct += correct
                total_samples += total

            # Calculate and print the accuracy for the epoch
        if (epoch + 1) % 30 == 0:
            epoch_accuracy = total_correct / total_samples
            print(f'Test Accuracy: {total_correct}/{total_samples}, {epoch_accuracy:.4f}')


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
    parser.add_argument('-d', '--dataset_name', type=str, default='Houston',#"Sao_Paulo"
                        help='Name of places')
    parser.add_argument('-g', '--group_number', type=int, default=0,
                        help='Number of groups')

    return parser.parse_args()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    layer_settings = list(itertools.product([0, 1, 2, 3], repeat=3))

    for layer_setting in layer_settings:

        net = end_to_end_Net_classification_small_Bayesian(n_in_channels=2, n_out_channels=2, n_classes=7, layer_setting=layer_setting)
        # net = end_to_end_Net_classification(n_in_channels=2, n_out_channels=2, n_classes=7, bilinear=True)

        logging.info(f'Network:\n'
                     f'\t{net.n_in_channels} input channels\n'
                     f'\t{net.n_classes} output channels (classes)\n'
                     )

        if args.load:
            net.load_state_dict(
                torch.load(args.load, map_location=device)
            )
            logging.info(f'Model loaded from {args.load}')

        net.to(device=device)
        # faster convolutions, but more memory
        # cudnn.benchmark = True

        n_train, train_loader, n_test, test_loader = dataset_func(dataset_name=args.dataset_name, batch_size=args.batchsize,
                      val_percent=args.val / 100, img_scale=args.scale)
        print('args.dataset_name:', args.dataset_name)
        print('args.batchsize:', args.batchsize)
        print('args.val / 100:', args.val / 100)
        print('args.scale:', args.scale)
        print('args.epochs:', args.epochs)

        try:
            train_net(net=net, device=device, dataset_name=args.dataset_name, n_train=n_train, train_loader=train_loader, test_loader=test_loader, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr,
                      val_percent=args.val / 100, img_scale=args.scale, layer_setting=layer_setting)

        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)


        # torch.save(net.state_dict(), 'path to save the trained model.pth')

