from numpy import genfromtxt
from metric_bandits.constants.paths import WINE_PATH

read_wine = genfromtxt(WINE_PATH, delimiter=',')

# WINE = data, labels
WINE = read_wine.T[1:14], read_wine.T[0]


