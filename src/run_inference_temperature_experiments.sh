#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

TOP_LEVEL_DIR=$(dirname "$parent_path")

# Directory containing the data files
INTRASENTENCE_DATA_PATH="${TOP_LEVEL_DIR}/data/df_intrasentence_de.pkl"

# Define softmax temperatures
TEMPERATURES=(0.5 2.0 2.5 3.0 5.0)

# Loop over different softmax temperatures
for temperature in "${TEMPERATURES[@]}"
do
    EXPERIMENT_ID="temperature_${temperature}"
    echo ">> Run Inference for temperature ${temperature}"
    python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path "$INTRASENTENCE_DATA_PATH" --experiment-id "$EXPERIMENT_ID" --softmax-temperature "$temperature"
done