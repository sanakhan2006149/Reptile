# Replica of BRR model from Reactive Sputtering study

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import config
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, explained_variance_score
#something ssrnejn
# Select size, dataset, output, and randomState from config
setSize = config.size
data = config.data
yIndex = config.yIndex
randomState = config.randomState
model = config.model

# Automating file creation
datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

directory = f"Saved Models/{datasetModels}/{output}/{model}"
os.makedirs(directory, exist_ok=True)
with open(f"Saved Models/{datasetModels}/{output}/{model}/{model} Metric Iteration Evaluation.txt", "w") as f:
    # Write headers
    f.write("MSE, RMSE, MAPE, EV, and R^2 Metrics\n")
    f.write(f"Current Model Dataset: {data}\n")
    f.write(f"Output Variable: {output}\n")
    f.write("=" * 50 + "\n")
    while setSize != 0:
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
        brr = BayesianRidge()
        brr.fit(xTrainScaled, yTrain)

        # Initial predictions
        yPredict = brr.predict(xTestScaled)
        mseCurrent = mean_squared_error(yTest, yPredict)
        rmseCurrent = np.sqrt(mseCurrent)
        mapeCurrent = np.mean(np.abs((yTest - yPredict) / yTest))
        evCurrent = explained_variance_score(yTest, yPredict)
        currentModelScore = brr.score(xTestScaled, yTest)

        # Write metrics
        f.write(f"Current Model Training Size: {setSize}\n")
        f.write(f"MSE: {mseCurrent}\n")
        f.write(f"RMSE: {rmseCurrent}\n")
        f.write(f"MAPE: {mapeCurrent}\n")
        f.write(f"EV: {evCurrent}\n")
        f.write(f"R^2: {currentModelScore}\n")
        f.write("-" * 50 + "\n")
        print(f"Completed {setSize}!")

        # Saving trained model
        directory = f"Saved Models/Starter Models/{datasetModels}/{output}/{model}/"
        modelName = f"{model.lower()}_model_{setSize}.pkl"
        os.makedirs(directory, exist_ok=True)
        joblib.dump(brr, os.path.join(directory, modelName))
        print("Saved!")
        setSize -= 5


# Plotting data
# plt.figure(figsize=(8, 8))
# sns.set(style="whitegrid")
# sns.scatterplot(x=yTest, y=yPredict, color="blue", s=50, edgecolor='black', alpha=0.75)
# min_val = min(min(yTest), min(yPredict))
# max_val = max(max(yTest), max(yPredict))
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit (y = x)")
# plt.title("BRR Model - " + ("Film-Thickness" if yIndex == -2 else "N/Ti Ratio"), fontsize=16)
# plt.xlabel("Measurements", fontsize=14)
# plt.ylabel("BRR Predictions", fontsize=14)
# plt.legend()
# plt.show()
