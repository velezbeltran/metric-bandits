from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from metric_bandits.constants.constants import SEED
import torch

n_samples = 60000

X, y = make_blobs(n_samples=n_samples, 
                centers=1, 
                n_features=2, 
                random_state=SEED
                )

scaler = StandardScaler()
X = scaler.fit_transform(X)

for i in range(n_samples):
    if X[i][0] < 0:
        y[i] = 1

BLOB = (torch.tensor(X), torch.tensor(y))  # this is the actual data



