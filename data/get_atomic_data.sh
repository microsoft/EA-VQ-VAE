# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz
mkdir -p atomic
mv atomic_data.tgz atomic

tar -xvzf atomic/atomic_data.tgz -C atomic
rm atomic/atomic_data.tgz
