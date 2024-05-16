#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

TOP_LEVEL_DIR=$(dirname "$parent_path")

# Directory containing the data files
INTRASENTENCE_DATA_PATH="${TOP_LEVEL_DIR}/data/df_intrasentence_de.pkl"


# Loop over ckpt files in saved_models directory
EXPERIMENT_ID="baseline"
echo ">> Run Inference for baseline model"
python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path "$INTRASENTENCE_DATA_PATH" --experiment-id "$EXPERIMENT_ID"