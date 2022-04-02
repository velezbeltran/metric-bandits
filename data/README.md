# Data
To generate the dasets run from the top level directory 
```
. scripts/make_data.sh
```
# Notes on datasets: 
## CIFAR 10
CIFAR 10: More information on the CIFAR 10 dataset can be found in `https://www.cs.toronto.edu/~kriz/cifar.html`.

## CIFAR 10 Triplets
The CIFAR 10 Triplets dataset can be created by running the `scripts/data/make_cifar_triplets.py` script. 
By default, it takes 5000 random samples and then creates triplets which are stored in batches of data. 
Each batch contains 10,000  `Triple` named tuples which have the two instances and the labels associated 
with these instances. See the file for more information. 
