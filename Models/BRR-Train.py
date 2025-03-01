# Training Loop for BRR model

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,  explained_variance_score

# Load BRR model
brr = joblib.load("Saved Models/Starter Models/brr_model.pkl")

# Saving first iteration
joblib.dump(brr, f"Saved Models/BRR-1/brr_model_iter_{1}.pkl")

# Selecting dataset
data = "Datasets/NitrideMetal (Dataset 2) NTi.csv"
df = pd.read_csv(data)

x = df.iloc[:, :-2].values
# Selecting output
# A y-index of -2 = film-thickness, -1 = N/Ti ratio
yIndex = -2
y = df.iloc[:, yIndex].values

# 80% data to train, 20% leave for testing. random_state is 40 initially
setSize = 40
trainSize = min(setSize, int(0.8 * len(x)), len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=40)

# Scaling data
xTrainLog = np.log1p(xTrain)
xTestLog = np.log1p(xTest)
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xTrainScaled = dataScaler.fit_transform(xTrainLog)
xTestScaled = dataScaler.transform(xTestLog)

# Initial predictions
yPredict = brr.predict(xTestScaled)
mseCurrent = mean_squared_error(yTest, yPredict)
rmseCurrent = np.sqrt(mseCurrent)
mapeCurrent = np.mean(np.abs((yTest - yPredict) / yTest))
evCurrent = explained_variance_score(yTest, yPredict)
currentModelScore = brr.score(xTestScaled, yTest)

with open("Saved Models/BRR-1/Metric Iteration Evaluation.txt", "w") as f:
    # Write headers
    f.write("MSE, RMSE, MAPE, EV, and R^2 Metrics\n")
    f.write(f"Current Model Dataset: {data}\n")
    f.write(f"Current Model Training Size: {setSize}\n")
    f.write("=" * 50 + "\n")
    f.write(f"Iteration 1 (Initial Model):\n")
    f.write(f"MSE: {mseCurrent}\n")
    f.write(f"RMSE: {rmseCurrent}\n")
    f.write(f"MAPE: {mapeCurrent}\n")
    f.write(f"EV: {evCurrent}\n")
    f.write(f"R^2: {currentModelScore}\n")
    f.write("-" * 50 + "\n")

    # Training model 19 times
    for i in range(1, 20):
        randomState = 40 + i
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

        # Scaling data
        xTrainLog = np.log1p(xTrain)
        xTestLog = np.log1p(xTest)
        dataScaler = MinMaxScaler(feature_range=(-1, 1))
        xTrainScaled = dataScaler.fit_transform(xTrainLog)
        xTestScaled = dataScaler.transform(xTestLog)

        # Fitting data
        brr.fit(xTrainScaled, yTrain)

        yPredict = brr.predict(xTestScaled)
        mseCurrent = mean_squared_error(yTest, yPredict)
        rmseCurrent = np.sqrt(mseCurrent)
        mapeCurrent = np.mean(np.abs((yTest - yPredict) / yTest))
        evCurrent = explained_variance_score(yTest, yPredict)
        currentModelScore = brr.score(xTestScaled, yTest)
        f.write(f"Iteration {i + 1}:\n")
        f.write(f"MSE: {mseCurrent}\n")
        f.write(f"RMSE: {rmseCurrent}\n")
        f.write(f"MAPE: {mapeCurrent}\n")
        f.write(f"EV: {evCurrent}\n")
        f.write(f"R^2: {currentModelScore}\n")
        f.write("-" * 50 + "\n")

        # Saving current model iteration
        joblib.dump(brr, f"Saved Models/BRR-1/brr_model_iter_{i + 1}.pkl")

print("Complete!")
