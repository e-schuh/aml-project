#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

echo "Generating underspecified examples..."
./generate_us_examples.sh

echo "Running preprocessing..."
./run_preprocessing.sh

echo "Training Models..."
./train_bert.sh