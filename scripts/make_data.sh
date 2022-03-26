#!/bin/bash

# Move to the base directory
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo "parent_path: $parent_path"
cd "$parent_path"


# Download the cifar 10 dataset
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P ./data/
tar -xzvf ./data/cifar-10-python.tar.gz -C ./data/
rm ./data/cifar-10-python.tar.gz
