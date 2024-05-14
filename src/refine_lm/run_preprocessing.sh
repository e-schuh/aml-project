#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

TOP_LEVEL_DIR=$(dirname $(dirname "$parent_path"))

# Directory containing the data files
DATA_DIR=${TOP_LEVEL_DIR}/data/refine_lm/training_data

# Path to the preprocess_data.py script
SCRIPT="preprocess_data.py"

# Loop over files ending with .source.json in the data directory
for file in "$DATA_DIR"/*.source.json; do
    # Check if the pattern matches any files
    if [[ -f $file ]]; then
        # Extract filename without the path
        filename=$(basename -- "$file")
        # Remove the extension and add .pkl
        output_filename="${filename%.source.json}.pkl"
        # Construct the full path for the output file
        output_file="$DATA_DIR/$output_filename"

        # Run the Python script with the appropriate arguments
        python3 "$SCRIPT" --input_path "$file" --out "$output_file"
    fi
done

echo "Data preprocessing completed."
