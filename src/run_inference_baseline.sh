#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

TOP_LEVEL_DIR=$(dirname "$parent_path")

# Directory containing the data files
INTRASENTENCE_DATA_PATH="${TOP_LEVEL_DIR}/data/df_intrasentence_de.pkl"


# Run SwissBERT baseline model
EXPERIMENT_ID="baseline"
echo ">> Run Inference for SwissBERT baseline model"
python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path "$INTRASENTENCE_DATA_PATH" --experiment-id "$EXPERIMENT_ID"

# Run BERT baseline model
EXPERIMENT_ID="baseline"
echo ">> Run Inference for BERT baseline model"
python3 -m inference --intrasentence-model "BertForMLM" --pretrained-model-name "bert-base-uncased" --intrasentence-data-path "$INTRASENTENCE_DATA_PATH" --experiment-id "$EXPERIMENT_ID"