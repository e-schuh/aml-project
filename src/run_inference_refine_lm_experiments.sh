#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

TOP_LEVEL_DIR=$(dirname "$parent_path")

# Directory containing the data files
DATA_DIR="${TOP_LEVEL_DIR}/data/refine_lm/saved_models"
INTRASENTENCE_DATA_PATH="${TOP_LEVEL_DIR}/data/df_intrasentence_de.pkl"


# Loop over ckpt files in saved_models directory
for file in ${DATA_DIR}/swissbert*.pt*; do
    EXPERIMENT_ID=$(echo "$file" | sed -n 's/.*_o_\(.*\)\.pt.*/\1/p')
    CKPT_FILE="$file"
    echo ">> Run Inference for model ckpt $CKPT_FILE"
    python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --ckpt-path "$CKPT_FILE" --intrasentence-data-path "$INTRASENTENCE_DATA_PATH" --experiment-id "$EXPERIMENT_ID"
done