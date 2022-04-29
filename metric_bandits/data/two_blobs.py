from sklearn.datasets import make_blobs
from metric_bandits.constants.constants import SEED
import torch

n_samples = 60000


X, y = make_blobs(n_samples=n_samples, 
                centers=[[-1,0],[1,0]],
                cluster_std=[[0.2, 2], [0.2,2]],
                n_features=2, 
                random_state=42
                  
                )

TWO_BLOBS = (torch.tensor(X), torch.tensor(y))  # this is the actual data



