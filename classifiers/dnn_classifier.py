import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class ClassifierNet(nn.Module):
    def __init__(self):
        self.input = nn.Linear(56, 150)
        self.hid1 = nn.Linear(150,150)
        self.out = nn.Linear(150,3)

    def forward(self, x):
        x = self.input(x)
        x = self.hid1(x)
        x = self.out(x)
        return x
