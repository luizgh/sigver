import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import Dict
from torch.nn.init import xavier_uniform_


class SigNetModel(nn.Module):
    """ SigNet model, from https://arxiv.org/abs/1705.05787
    """
    def __init__(self, normalize):
        super(SigNetModel, self).__init__()
        self.feature_space_size = 2048
        self.normalize = normalize

        if normalize:
            self.bn1 = torch.nn.BatchNorm2d(96, affine=False)
            self.bn2 = torch.nn.BatchNorm2d(256, affine=False)
            self.bn3 = torch.nn.BatchNorm2d(384, affine=False)
            self.bn4 = torch.nn.BatchNorm2d(384, affine=False)
            self.bn5 = torch.nn.BatchNorm2d(256, affine=False)

            self.bn6 = torch.nn.BatchNorm1d(2048, affine=False)
            self.bn7 = torch.nn.BatchNorm1d(2048, affine=False)
        else:
            self.bn1 = self.bn2 = self.bn3 = self.bn4 = self.bn5 = self.bn6 = self.bn7 = None

    def build_weights(self, device: torch.device) -> Dict[str, torch.tensor]:
        weights = OrderedDict([
            ['conv1', xavier_uniform_(torch.empty((96, 1, 11, 11), requires_grad=True, device=device))],
            ['b1', torch.zeros(96, requires_grad=True, device=device)],
            ['conv2', xavier_uniform_(torch.empty((256, 96, 5, 5), requires_grad=True, device=device))],
            ['b2', torch.zeros(256, requires_grad=True, device=device)],
            ['conv3', xavier_uniform_(torch.empty((384, 256, 3, 3), requires_grad=True, device=device))],
            ['b3', torch.zeros(384, requires_grad=True, device=device)],
            ['conv4', xavier_uniform_(torch.empty((384, 384, 3, 3), requires_grad=True, device=device))],
            ['b4', torch.zeros(384, requires_grad=True, device=device)],
            ['conv5', xavier_uniform_(torch.empty((256, 384, 3, 3), requires_grad=True, device=device))],
            ['b5', torch.zeros(256, requires_grad=True, device=device)],

            ['w6', xavier_uniform_(torch.empty((self.feature_space_size, 256 * 3 * 5), requires_grad=True, device=device))],
            ['b6', torch.zeros(self.feature_space_size, requires_grad=True, device=device)],

            ['w7', xavier_uniform_(torch.empty((self.feature_space_size, self.feature_space_size), requires_grad=True, device=device))],
            ['b7', torch.zeros(self.feature_space_size, requires_grad=True, device=device)],

            ['w8', xavier_uniform_(torch.empty((1, self.feature_space_size), requires_grad=True, device=device))],
            ['b8', torch.zeros(1, requires_grad=True, device=device)],
        ])
        return weights

    def conv_bn_relu(self, input, weights, bias, bn_layer, training, stride=1, pad=0):
        x = input
        if self.normalize:
            x = F.conv2d(x, weights, bias=None, stride=stride, padding=pad)
            bn_layer.train(training)
            x = bn_layer(x) + bias.view(1, -1, 1, 1)
        else:
            x = F.conv2d(x, weights, bias, stride=stride, padding=pad)
        x = F.relu(x)
        return x

    def dense_bn_relu(self, input, weights, bias, bn_layer, training):
        x = input
        if self.normalize:
            x = F.linear(x, weights, bias=None)
            bn_layer.train(training)
            x = bn_layer(x) + bias.view(1, -1)
        else:
            x = F.linear(x, weights, bias)
        x = F.relu(x)
        return x

    def get_features(self, inputs, weights, training):
        x = inputs

        x = self.conv_bn_relu(x, weights['conv1'], weights['b1'], self.bn1, training, stride=4)
        x = F.max_pool2d(x, 3, 2)
        x = self.conv_bn_relu(x, weights['conv2'], weights['b2'], self.bn2, training, pad=2)
        x = F.max_pool2d(x, 3, 2)
        x = self.conv_bn_relu(x, weights['conv3'], weights['b3'], self.bn3, training, pad=1)
        x = self.conv_bn_relu(x, weights['conv4'], weights['b4'], self.bn4, training, pad=1)
        x = self.conv_bn_relu(x, weights['conv5'], weights['b5'], self.bn5, training, pad=1)
        x = F.max_pool2d(x, 3, 2)

        x = x.view(x.shape[0], 256 * 3 * 5)
        x = self.dense_bn_relu(x, weights['w6'], weights['b6'], self.bn6, training)
        x = self.dense_bn_relu(x, weights['w7'], weights['b7'], self.bn7, training)
        return x

    def forward(self, inputs, weights, training):
        x = self.get_features(inputs, weights, training)
        x = F.linear(x, weights['w8'], weights['b8'])
        return x


