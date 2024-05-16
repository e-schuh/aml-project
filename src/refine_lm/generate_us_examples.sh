#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
TOP_LEVEL_DIR=$(dirname "$(dirname "$parent_path")")
TRAINING_DATA_DIR="${TOP_LEVEL_DIR}/data/refine_lm/training_data"
mkdir -p ${TRAINING_DATA_DIR}

# Generate mixed gender data

TYPE=slot_act_map
SUBJ=mixed_gender_bert_test
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
MODEL=distilbert-base-uncased
python3 -m templates.generate_underspecified_templates --template_type $TYPE \
--subj $SUBJ --act $ACT --slot $SLOT \
--output "${TRAINING_DATA_DIR}/${FILE}.source.json"


TYPE=slot_act_map
SUBJ=mixed_gender_bert_train
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
MODEL=distilbert-base-uncased
python3 -m templates.generate_underspecified_templates --template_type $TYPE \
--subj $SUBJ --act $ACT --slot $SLOT \
--output "${TRAINING_DATA_DIR}/${FILE}.source.json"



# Generate training data for other bias categories

for DATA in country religion ethnicity; do
    TYPE=slot_act_map
    SUBJ=${DATA}_bert
    SLOT=${DATA}_noact_lm-TEST
    ACT=biased_${DATA}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    python3 -m templates.generate_underspecified_templates --template_type $TYPE \
    --subj $SUBJ --act $ACT --slot $SLOT \
    --output "${TRAINING_DATA_DIR}/${FILE}.source.json"
done


for DATA in country religion ethnicity; do
    TYPE=slot_act_map
    SUBJ=${DATA}_bert
    SLOT=${DATA}_noact_lm-TRAIN
    ACT=biased_${DATA}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    python3 -m templates.generate_underspecified_templates --template_type $TYPE \
    --subj $SUBJ --act $ACT --slot $SLOT \
    --output "${TRAINING_DATA_DIR}/${FILE}.source.json"
done