# ========================================

# WriteMetrics, DataGenerate, and GridSearch Configuration
size: int = 40
data: str = "Datasets/Nitride (Dataset 1) NTi.csv"
randomState: int = 47
yIndex: int = -2

# (-2) = film-thickness
# (-1) = N/Ti ratio

# ========================================

# Neural Network Configuration
nSize: int = 40
nData: str = "Datasets/Nitride (Dataset 1) NTi.csv"
nRandomState: int = 47
nYIndex: int = -2
epochs: int = 16
batchSize: int = 20
learningRate: float = 0.001

# (-2) = film-thickness
# (-1) = N/Ti ratio
# Pretraining Batch Size = {16, 512, 1028}
# Pretraining Epoch = {20, 200, 1000}

# ========================================


# AVAILABLE DATASETS:
    # Datasets/FullData.csv
    # Datasets/Metal (Alone).csv
    # Datasets/Metal (Alone) NTi.csv
    # Datasets/Nitride (Dataset 1).csv
        # Datasets/Nitride (Dataset 1) NTi.csv
    # Datasets/NitrideMetal (Dataset 2).csv
        # Datasets/NitrideMetal (Dataset 2) NTi.csv