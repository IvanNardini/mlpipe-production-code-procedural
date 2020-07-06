#!/bin/bash

cd ./procedural_ml_pipe

for script in $*; do
    if [ $script == 'train' ]; then
        python3 train.py
    fi
    if [ $script == 'score' ]; then
        python3 score.py
    fi
done