from numpy import genfromtxt
from metric_bandits.constants.paths import WINE_PATH
import numpy as np

read_wine = genfromtxt(WINE_PATH, delimiter=',')

# WINE = data, labels
features = np.array(read_wine.T[1:14]).T
labels = np.array(read_wine.T[0])

WINE = list(zip(features,labels))



