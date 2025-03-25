# Pretraining Fully Connected Neural Network Model using SVG augmented data
# Pretraining Batch Size = {16, 512, 1028}
# Pretraining Epoch = {20, 200, 1000}

import numpy as np
import copy
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from typing import Callable
import pandas as pd
import config

def constructModelLayers():
    model = Sequential()
    model.add(Dense(4, input_dim=8, activation='relu')) # 8 input nodes, into Hidden Layer 1
    model.add(Dense(4, activation='relu')) # Hidden Layer 2
    model.add(Dense(1, activation='linear')) # Output Layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Select size, dataset, output, and randomState from config
setSize = config.nSize
yIndex = config.nYIndex
randomState = config.nRandomState
data = config.nData
model = "SVR"
datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

# Import Augmented Data CSV file
augmentedData = pd.read_csv(f"Saved Models/{datasetModels}/{output}/{model}/{model} Size_{setSize} Random_{randomState} Augmented Data.csv")

# Construct FCNN Model
fcnnModel = constructModelLayers()

# Normalize data

# Pre-train FCNN

# Save FCNN