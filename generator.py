import argparse
import os
import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
from torch.utils.tensorboard import writer, SummaryWriter
from torch.utils.data import DataLoader
from math import pi
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def block(in_feat, n_filters, kernel_size, padding, upsample_size):
            layers = []
            layers.append(nn.Conv1d(in_feat, n_filters, kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(n_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Upsample(upsample_size))
            return layers
        
        self.model = nn.Sequential(
            nn.Linear(50, 100),
            nn.BatchNorm1d(100, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ExpandDimension(),
            *block(1, n_filters=32, kernel_size=3, padding=1, upsample_size=200),
            *block(32, n_filters=32, kernel_size=3, padding=1, upsample_size=400),
            *block(32, n_filters=32, kernel_size=3, padding=1, upsample_size=800),
            nn.Conv1d(32, 1, 3, padding=1),
            nn.BatchNorm1d(1, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            SqueezeDimension(),
            nn.Linear(800, 100),
            nn.BatchNorm1d(100, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)