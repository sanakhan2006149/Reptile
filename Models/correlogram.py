# Making correlogram of input and output variables

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Selecting dataset
df = pd.read_csv("Datasets/NitrideMetal (Dataset 2) NTi.csv")

# Creating correlogram
correlationMatrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlogram of Variables')
plt.show()
