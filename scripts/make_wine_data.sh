#!/bin/bash

# Move to the base directory
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo "parent_path: $parent_path"
cd "$parent_path"


# Download the wine dataset. 
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data -P ./data/

