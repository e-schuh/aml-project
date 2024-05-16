#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

TOP_LEVEL_DIR=$(dirname "$(dirname "$parent_path")")

# Directory containing the data files
DATA_DIR="${TOP_LEVEL_DIR}/data/concept_eraser/hidden_states"

## CONCEPT=gender; after language adapters ##
CONCEPT="gender"
PATH_TO_HIDDEN_STATES="${DATA_DIR}/dev_gender_cls.pt"
PATH_TO_CONCEPT_LABELS="${DATA_DIR}/dev_gender_labels.pt"
ERASER_ID="genderAfterLangAdp"
echo ">> Training concept eraser model for hidden states after language adapter for concept "${CONCEPT}
python3 -m train_concept_erasure_model --path-to-hidden-states ${PATH_TO_HIDDEN_STATES} --path-to-concept-labels ${PATH_TO_CONCEPT_LABELS} --experiment-id ${ERASER_ID}


## CONCEPT=gender; before language adapters ##
CONCEPT="gender"
PATH_TO_HIDDEN_STATES="${DATA_DIR}/dev_gender_cls_beforeLangAdp.pt"
PATH_TO_CONCEPT_LABELS="${DATA_DIR}/dev_gender_labels_beforeLangAdp.pt"
ERASER_ID="genderBeforeLangAdp"
echo ">> Training concept eraser model for hidden states after language adapter for concept "${CONCEPT}
python3 -m train_concept_erasure_model --path-to-hidden-states ${PATH_TO_HIDDEN_STATES} --path-to-concept-labels ${PATH_TO_CONCEPT_LABELS} --experiment-id ${ERASER_ID}


## CONCEPT=profession; after language adapters ##
CONCEPT="profession"
PATH_TO_HIDDEN_STATES="${DATA_DIR}/dev_profession_cls.pt"
PATH_TO_CONCEPT_LABELS="${DATA_DIR}/dev_profession_labels.pt"
ERASER_ID="professionAfterLangAdp"
echo ">> Training concept eraser model for hidden states after language adapter for concept "${CONCEPT}
python3 -m train_concept_erasure_model --path-to-hidden-states ${PATH_TO_HIDDEN_STATES} --path-to-concept-labels ${PATH_TO_CONCEPT_LABELS} --experiment-id ${ERASER_ID}


## CONCEPT=profession; before language adapters ##
CONCEPT="profession"
PATH_TO_HIDDEN_STATES="${DATA_DIR}/dev_profession_cls_beforeLangAdp.pt"
PATH_TO_CONCEPT_LABELS="${DATA_DIR}/dev_profession_labels_beforeLangAdp.pt"
ERASER_ID="professionBeforeLangAdp"
echo ">> Training concept eraser model for hidden states after language adapter for concept "${CONCEPT}
python3 -m train_concept_erasure_model --path-to-hidden-states ${PATH_TO_HIDDEN_STATES} --path-to-concept-labels ${PATH_TO_CONCEPT_LABELS} --experiment-id ${ERASER_ID}