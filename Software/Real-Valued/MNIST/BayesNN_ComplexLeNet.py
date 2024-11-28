import sys
sys.path.append('../../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexMaxPool2d, ComplexReLU
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from utils_mnist import evaluate
from complex_dropout import Complex_BernoulliDropout
import itertools
import argparse


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        if len(x.shape) == 1:
            return x.unsqueeze(dim=0)
        return x.reshape(x.size(0), -1)  # x: torch.Size([256, 512, 1, 1])


# class ComplexNet_LeNet(nn.Module):
#     def __init__(self, init_channels, output_size, layer_setting, dropout_rate=0.25):
#         super(ComplexNet_LeNet, self).__init__()
#         self.init_channels = init_channels
#         self.output_size = output_size
#         self.layers = nn.ModuleList([
#             ComplexConv2d(in_channels=self.init_channels, out_channels=20, kernel_size=5, padding=2, bias=False),
#             ComplexMaxPool2d(kernel_size=2, stride=2),
#             Complex_BernoulliDropout(dropout_rate, real_imag=layer_setting[0]),
#
#             ComplexConv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2, bias=False),
#             ComplexMaxPool2d(kernel_size=2, stride=2),
#             Complex_BernoulliDropout(dropout_rate, real_imag=layer_setting[1]),
#
#             Flatten(),
#             ComplexLinear(in_features=50 * 7 * 7, out_features=500, bias=False),
#             ComplexReLU(),
#             Complex_BernoulliDropout(dropout_rate, real_imag=layer_setting[2]),
#
#             ComplexLinear(in_features=500, out_features=self.output_size, bias=False)
#         ])
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         x = x.abs()
#         x = F.log_softmax(x, dim=1)
#         return x



class ComplexNet_LeNet(nn.Module):
    def __init__(self, init_channels, output_size, layer_setting, dropout_rate=0.25):
        super(ComplexNet_LeNet, self).__init__()
        self.init_channels = init_channels
        self.output_size = output_size
        self.layers = nn.ModuleList([
            ComplexConv2d(in_channels=self.init_channels, out_channels=6, kernel_size=5, padding=2),
            ComplexBatchNorm2d(6),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=2, stride=2),
            Complex_BernoulliDropout(dropout_rate, real_imag=layer_setting[0]),

            ComplexConv2d(in_channels=6, out_channels=16, kernel_size=5, padding=2),
            ComplexBatchNorm2d(16),
            ComplexReLU(),
            ComplexMaxPool2d(kernel_size=2, stride=2),
            Complex_BernoulliDropout(dropout_rate, real_imag=layer_setting[1]),

            Flatten(),
            ComplexLinear(in_features=16 * 7 * 7, out_features=84),
            ComplexReLU(),
            Complex_BernoulliDropout(dropout_rate, real_imag=layer_setting[2]),
            ComplexLinear(in_features=84, out_features=self.output_size)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.abs()
        x = F.log_softmax(x, dim=1)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward() #[W Copy.cpp:244] Warning: Casting complex values to real discards the imaginary part (function operator())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )

def test(model, device, test_loader, samples_number=1):
    model.eval()

    outputs = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).type(torch.complex64), target.to(device)

            samples = []
            for j in range(samples_number):
                output = model(data)
                samples.append(output)
            outputs.append(torch.stack(samples, dim=1).mean(dim=1))
            targets.append(target)

        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)

        error, ece, entropy, loss = evaluate(output=outputs, target=targets) # outputs:torch.Size([10000, 10])targets:torch.Size([10000])
        print('acc, ece, loss:', 100 - error, ece, loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default= True, help='training')
    parser.add_argument('--testing', default= False, help='testing')
    parser.add_argument('--sampling_number', default=3, help='sampling number')
    args = parser.parse_args()

    batch_size = 256
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
    test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    layer_settings = list(itertools.product([1, 2, 3], repeat=3))
    if args.training:
        for layer_setting in layer_settings:
            print('layer_setting:', layer_setting)
            model = ComplexNet_LeNet(init_channels=1, output_size=10, layer_setting=layer_setting).to(device)

            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            model.train()
            for epoch in range(10):
                # print('epoch:', epoch, '----------------------------------------------------------')
                train(model, device, train_loader, optimizer, epoch)
            torch.save(model, f'./BayesCVLeNet_trained_model/complex_{layer_setting}.pth')

    if args.testing:
        for layer_setting in layer_settings:
            model = torch.load(f'./BayesCVLeNet_trained_model/complex_{layer_setting}.pth')
            print(layer_setting,':')
            test(model, device, test_loader, samples_number=args.sampling_number)



