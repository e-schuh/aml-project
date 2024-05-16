#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

TOP_LEVEL_DIR=$(dirname "$(dirname "$parent_path")")

# Directory containing the data files
DATA_DIR="${TOP_LEVEL_DIR}/data/refine_lm/training_data"



## TOPK=8, EPOCHS=1 ##
topk=8
cat=gender
epochs=1
model=swissbert_o_${cat}_tk${topk}_ep${epochs}
echo ">> Training model "${model}
python3 -m training_bert --ppdata "${DATA_DIR}/slotmap_mixedgenderberttrain_occupationrev1_gendernoactlm.pkl" --topk ${topk} --model_name ${model} --epochs ${epochs}


## TOPK=20, EPOCHS=1 ##
topk=20
cat=gender
epochs=1
model=swissbert_o_${cat}_tk${topk}_ep${epochs}
echo ">> Training model "${model}
python3 -m training_bert --ppdata "${DATA_DIR}/slotmap_mixedgenderberttrain_occupationrev1_gendernoactlm.pkl" --topk ${topk} --model_name ${model} --epochs ${epochs}


## TOPK=8, EPOCHS=2 ##
topk=8
cat=gender
epochs=2
model=swissbert_o_${cat}_tk${topk}_ep${epochs}
echo ">> Training model "${model}
python3 -m training_bert --ppdata "${DATA_DIR}/slotmap_mixedgenderberttrain_occupationrev1_gendernoactlm.pkl" --topk ${topk} --model_name ${model}


## TOPK=20, EPOCHS=2 ##
topk=20
cat=gender
epochs=2
model=swissbert_o_${cat}_tk${topk}_ep${epochs}
echo ">> Training model "${model}
python3 -m training_bert --ppdata "${DATA_DIR}/slotmap_mixedgenderberttrain_occupationrev1_gendernoactlm.pkl" --topk ${topk} --model_name ${model}

exit 0