from collections.__init__ import OrderedDict
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.feature_space_size = 256

    def build_weights(self, device: torch.device) -> Dict[str, torch.tensor]:
        weights = OrderedDict([
            ['conv1', xavier_uniform_(torch.empty((32, 1, 5, 5), requires_grad=True, device=device))],
            ['b1', torch.zeros(32, requires_grad=True, device=device)],
            ['conv2', xavier_uniform_(torch.empty((32, 32, 5, 5), requires_grad=True, device=device))],
            ['b2', torch.zeros(32, requires_grad=True, device=device)],

            ['w3', xavier_uniform_(torch.empty((1024, 32 * 5 * 7), requires_grad=True, device=device))],
            ['b3', torch.zeros(1024, requires_grad=True, device=device)],

            ['w4', xavier_uniform_(torch.empty((self.feature_space_size, 1024), requires_grad=True, device=device))],
            ['b4', torch.zeros(self.feature_space_size, requires_grad=True, device=device)],

            ['w5', xavier_uniform_(torch.empty((1, self.feature_space_size), requires_grad=True, device=device))],
            ['b5', torch.zeros(1, requires_grad=True, device=device)],
        ])

        return weights

    def get_features(self, inputs, weights, training):
        x = inputs
        x = F.conv2d(x, weights['conv1'], weights['b1'])
        x = F.relu(x)
        x = F.max_pool2d(x, 5, 5)
        x = F.conv2d(x, weights['conv2'], weights['b2'])
        x = F.relu(x)
        x = F.max_pool2d(x, 5, 5)
        x = x.view(x.shape[0], 32 * 5 * 7)

        x = F.linear(x, weights['w3'], weights['b3'])
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=training)
        x = F.linear(x, weights['w4'], weights['b4'])
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=training)
        return x

    def forward(self, inputs, weights, training):
        x = self.get_features(inputs, weights, training)
        x = F.linear(x, weights['w5'], weights['b5'])
        return x