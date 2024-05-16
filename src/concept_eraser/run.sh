#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

echo "Computing hidden states for concept eraser training..."
./run_get_hidden_states.sh

echo "Training of concept eraser models..."
./run_train_concept_eraser_variations.sh