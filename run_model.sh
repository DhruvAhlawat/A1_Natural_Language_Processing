# !/bin/bash

if [ "$1" = "test" ]; then
    python3 test_model.py $2 $3 $4
elif [ "$1" = "train" ]; then
    python3 train_model.py $2 $3
elif [ "$1" = "clean" ]; then
    rm -rf $2
else
    echo "Invalid option"
fi