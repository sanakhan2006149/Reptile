# Displaying Graph and Metrics for a given BRR model

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,  explained_variance_score

# Load BRR model
brr = joblib.load("Saved Models/BRR-1/brr_model_iter_20.pkl")

# Selecting dataset
data = "Datasets/NitrideMetal (Dataset 2) NTi.csv"
df = pd.read_csv(data)

x = df.iloc[:, :-2].values
# Selecting output
# A y-index of -2 = film-thickness, -1 = N/Ti ratio
yIndex = -2
y = df.iloc[:, yIndex].values

# 80% data to train, 20% leave for testing. random_state is 40 temporarily,
# Will look for the study's random_state later
setSize = 40
trainSize = min(setSize, int(0.8 * len(x)), len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=2)

# Scaling data
xTrainLog = np.log1p(xTrain)
xTestLog = np.log1p(xTest)
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xTrainScaled = dataScaler.fit_transform(xTrainLog)
xTestScaled = dataScaler.transform(xTestLog)

# BRR model making predictions
yPredict = brr.predict(xTestScaled)
mseCurrent = mean_squared_error(yTest, yPredict)
rmseCurrent = np.sqrt(mseCurrent)
mapeCurrent = np.mean(np.abs((yTest - yPredict) / yTest))
evCurrent = explained_variance_score(yTest, yPredict)
currentModelScore = brr.score(xTestScaled, yTest)
print("Current Model Dataset:", data)
print("Current Model Training Size:", setSize)
print("Current Model MSE:", mseCurrent)
print("Current Model RMSE:", rmseCurrent)
print("Current Model MAPE:", mapeCurrent)
print("Current Model EV:", evCurrent)
print("Current Model R^2:", currentModelScore)

# Plotting data
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")
sns.scatterplot(x=yTest, y=yPredict, color="blue", s=50, edgecolor='black', alpha=0.75)
min_val = min(min(yTest), min(yPredict))
max_val = max(max(yTest), max(yPredict))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit (y = x)")
plt.title("BRR Model - " + ("Film-Thickness" if yIndex == -2 else "N/Ti Ratio"), fontsize=16)
plt.xlabel("Measurements", fontsize=14)
plt.ylabel("BRR Predictions", fontsize=14)
plt.legend()
plt.show()