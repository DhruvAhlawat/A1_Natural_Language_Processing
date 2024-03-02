# !/bin/bash

if [ "$1" = "test" ]; then
    python3 test_model.py $2 $3 $4
else
    python3 train_model.py $2 $3
fi