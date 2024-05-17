#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

TOP_LEVEL_DIR=$(dirname "$parent_path")

# Directory containing the ground truth file
FILE_PATH_GT="${TOP_LEVEL_DIR}/data/df_intrasentence_de.pkl"

# Output path
OUTPUT_PATH="${TOP_LEVEL_DIR}/evaluation_output/eval_metrics_SwissBert_temperatures_de.json"

# Loop over files in inference_output directory
for file in ${TOP_LEVEL_DIR}/inference_output/combined_results*_temperature*; do
    FILE_PATH_PRED="$file"
    echo ">> Evaluating $FILE_PATH_PRED"
    python3 -m evaluation --intrasentence-gold-file-path "$FILE_PATH_GT" --inference-output-file "$FILE_PATH_PRED" --skip-intersentence --output-file "$OUTPUT_PATH"
done