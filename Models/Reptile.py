import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Callable
import matplotlib.pyplot as plt
import pandas as pd


class Reptile(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def pretrain(self, SVRdataFile: str, trainCount: int):
        with open(SVRdataFile, "r") as f:
            data = f.readline()
            df = pd.read_csv(data)
            x = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values




def reptile(model: nn.Module, nb_interations: int, sample_task: Callable,
            perform_k_training_steps: Callable, k=1, epsilon = 0.1):
    for _ in tqdm(range(nb_interations)):
        task = sample_task()
        phi_tilde = perform_k_training_steps(copy.deepcopy(model), task, k)
        with torch.no_grad():
            for p, g in zip(model.parameters(), phi_tilde):
                p += epsilon * (g-p)