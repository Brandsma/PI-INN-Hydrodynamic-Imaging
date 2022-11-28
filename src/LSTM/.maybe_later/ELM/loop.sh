#!/bin/bash
fold_vals=(1 2 3 4 5)

for fold_val in "${fold_vals[@]}"
do
    python train_mnist.py 30000 15 1 $fold_val
done