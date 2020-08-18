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
from torch.utils.data import Dataset, DataLoader

data = pd.read_csv('./data/SPY.csv')

class Returns(Dataset):
    def __init__(self, data_frame, window_size):
        self.data = data_frame.values
        self.window_size = window_size
        self.dataset = self._generate_dataset()

    def __len__(self):
        return self.data.shape[0] // self.window_size

    def __getitem__(self, index):
        return self.dataset[index]

    def _generate_dataset(self):
        dataset = []
        for i in range(2 * self.__len__() - 1):
            sample = self.data[int(i * self.window_size / 2): int(i * self.window_size / 2) + self.window_size]
            sample = np.squeeze(sample)
            dataset.append(sample)

        return np.array(dataset)