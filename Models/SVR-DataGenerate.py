# Generating augmented data - SVR

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import config
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, explained_variance_score

# Select size, dataset, output, and randomState from config
setSize = config.size
data = config.data
yIndex = config.yIndex
randomState = config.randomState
model = "SVR"

# Automating file creation
datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

directory = f"Saved Models/{datasetModels}/{output}/{model}"
os.makedirs(directory, exist_ok=True)
df = pd.read_csv(data)
x = df.iloc[:, :-2].values
# Selecting output
y = df.iloc[:, yIndex].values

# 80% data to train, 20% leave for testing. random_state is set in config
trainSize = min(setSize, int(0.8 * len(x)), len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

# Scaling data
xTrainLog = np.log1p(xTrain)
xTestLog = np.log1p(xTest)
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xTrainScaled = dataScaler.fit_transform(xTrainLog)
xTestScaled = dataScaler.transform(xTestLog)

# Init BRR model
svr = SVR(kernel='rbf', C=1000.0, epsilon=0.5, gamma='scale')
svr.fit(xTrainScaled, yTrain)

xMin, xMax = xTrain.min(axis=0), xTrain.max(axis=0)

# Interpolation
xInterpolated = np.linspace(xMin, xMax, num=822)
xInterpolatedLog = np.log1p(xInterpolated)
xInterpolatedScaled = dataScaler.transform(xInterpolatedLog)
yInterpolated = svr.predict(xInterpolatedScaled)
interpolatedData = np.hstack((xInterpolated, yInterpolated.reshape(-1, 1)))

# Extrapolation
xExtrapolatedMin = xMin - 0.1 * (xMax - xMin)
xExtrapolatedMax = xMax + 0.1 * (xMax - xMin)
xExtrapolated = np.linspace(xExtrapolatedMin, xExtrapolatedMax, num=206)
xExtrapolatedLog = np.log1p(xExtrapolated)
xExtrapolatedScaled = dataScaler.transform(xExtrapolatedLog)
yExtrapolated = svr.predict(xExtrapolatedScaled)
extrapolatedData = np.hstack((xExtrapolated, yExtrapolated.reshape(-1, 1)))

directory = f"Saved Models/{datasetModels}/{output}/{model}/"
os.makedirs(directory, exist_ok=True)
np.savetxt(f"{directory}interpolated_data.csv", interpolatedData, delimiter=',')
np.savetxt(f"{directory}/extrapolated_data.csv", extrapolatedData, delimiter=',')




