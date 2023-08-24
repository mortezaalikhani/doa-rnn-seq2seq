"""
GRU classifier

"""

import random

import numpy as np
import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, dense_size, hid_dim, num_layer, num_classes, dropout, seed):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.fc1 = nn.Linear(input_dim, dense_size)

        self.relu = nn.ReLU()

        self.rnn = nn.GRU(dense_size, hid_dim, num_layers=num_layer, bidirectional=False)

        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hid_dim, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        x = self.fc1(src)

        x = self.relu(x)

        x, _ = self.rnn(x)

        x = x[-1, :, :]

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        return x
