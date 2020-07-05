#!/bin/bash

cd ./procedural_ml_pipe

for script in $*; do
    python3 $script.py
done
