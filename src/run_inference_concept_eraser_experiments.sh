#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

TOP_LEVEL_DIR=$(dirname "$parent_path")

# Directory containing the data files
DATA_DIR="${TOP_LEVEL_DIR}/data/concept_eraser/eraser_models"
INTRASENTENCE_DATA_PATH="${TOP_LEVEL_DIR}/data/df_intrasentence_de.pkl"


#### Application of one concept eraser each ####

## CONCEPT=gender; after language adapters ##
ERASER_ID="genderAfterLangAdp"
CONCEPT="gender"
PATH_TO_ERASER="${DATA_DIR}/eraser_${ERASER_ID}.pkl"
echo ">> Inference with concept eraser after language adapter for concept "${CONCEPT}
python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path ${INTRASENTENCE_DATA_PATH} --experiment-id ${ERASER_ID} --eraser-path-list ${PATH_TO_ERASER}

## CONCEPT=gender; before language adapters ##
ERASER_ID="genderBeforeLangAdp"
CONCEPT="gender"
PATH_TO_ERASER="${DATA_DIR}/eraser_${ERASER_ID}.pkl"
echo ">> Inference with concept eraser before language adapter for concept "${CONCEPT}
python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path ${INTRASENTENCE_DATA_PATH} --experiment-id ${ERASER_ID} --eraser-path-list ${PATH_TO_ERASER} --eraser-before-lang-adapt

## CONCEPT=profession; after language adapters ##
ERASER_ID="professionAfterLangAdp"
CONCEPT="profession"
PATH_TO_ERASER="${DATA_DIR}/eraser_${ERASER_ID}.pkl"
echo ">> Inference with concept eraser after language adapter for concept "${CONCEPT}
python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path ${INTRASENTENCE_DATA_PATH} --experiment-id ${ERASER_ID} --eraser-path-list ${PATH_TO_ERASER}

## CONCEPT=profession; before language adapters ##
ERASER_ID="professionBeforeLangAdp"
CONCEPT="profession"
PATH_TO_ERASER="${DATA_DIR}/eraser_${ERASER_ID}.pkl"
echo ">> Inference with concept eraser before language adapter for concept "${CONCEPT}
python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path ${INTRASENTENCE_DATA_PATH} --experiment-id ${ERASER_ID} --eraser-path-list ${PATH_TO_ERASER} --eraser-before-lang-adapt



#### Sequential application of two concept erasers ####

## CONCEPT1=gender and CONCEPT2=profession; after language adapters ##
ERASER_ID="genderProfessionAfterLangAdp"
ERASER_ID1="genderAfterLangAdp"
ERASER_ID2="professionAfterLangAdp"
CONCEPT1="gender"
CONCEPT2="profession"
PATH_TO_ERASER1="${DATA_DIR}/eraser_${ERASER_ID1}.pkl"
PATH_TO_ERASER2="${DATA_DIR}/eraser_${ERASER_ID2}.pkl"
echo ">> Inference with concept eraser after language adapter for concepts: "${CONCEPT1}" and "${CONCEPT2}""
python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path ${INTRASENTENCE_DATA_PATH} --experiment-id ${ERASER_ID} --eraser-path-list ${PATH_TO_ERASER1} ${PATH_TO_ERASER2}

## CONCEPT1=gender and CONCEPT2=profession; before language adapters ##
ERASER_ID="genderProfessionBeforeLangAdp"
ERASER_ID1="genderBeforeLangAdp"
ERASER_ID2="professionBeforeLangAdp"
CONCEPT1="gender"
CONCEPT2="profession"
PATH_TO_ERASER1="${DATA_DIR}/eraser_${ERASER_ID1}.pkl"
PATH_TO_ERASER2="${DATA_DIR}/eraser_${ERASER_ID2}.pkl"
echo ">> Inference with concept eraser before language adapter for concepts: "${CONCEPT1}" and "${CONCEPT2}""
python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path ${INTRASENTENCE_DATA_PATH} --experiment-id ${ERASER_ID} --eraser-path-list ${PATH_TO_ERASER1} ${PATH_TO_ERASER2} --eraser-before-lang-adapt

## CONCEPT2=profession and CONCEPT1=gender; after language adapters ##
ERASER_ID="professionGenderAfterLangAdp"
ERASER_ID1="genderAfterLangAdp"
ERASER_ID2="professionAfterLangAdp"
CONCEPT1="gender"
CONCEPT2="profession"
PATH_TO_ERASER1="${DATA_DIR}/eraser_${ERASER_ID1}.pkl"
PATH_TO_ERASER2="${DATA_DIR}/eraser_${ERASER_ID2}.pkl"
echo ">> Inference with concept eraser after language adapter for concepts: "${CONCEPT2}" and "${CONCEPT1}""
python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path ${INTRASENTENCE_DATA_PATH} --experiment-id ${ERASER_ID} --eraser-path-list ${PATH_TO_ERASER2} ${PATH_TO_ERASER1}

## CONCEPT2=profession and CONCEPT1=gender; before language adapters ##
ERASER_ID="professionGenderBeforeLangAdp"
ERASER_ID1="genderBeforeLangAdp"
ERASER_ID2="professionBeforeLangAdp"
CONCEPT1="gender"
CONCEPT2="profession"
PATH_TO_ERASER1="${DATA_DIR}/eraser_${ERASER_ID1}.pkl"
PATH_TO_ERASER2="${DATA_DIR}/eraser_${ERASER_ID2}.pkl"
echo ">> Inference with concept eraser before language adapter for concepts: "${CONCEPT2}" and "${CONCEPT1}""
python3 -m inference --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path ${INTRASENTENCE_DATA_PATH} --experiment-id ${ERASER_ID} --eraser-path-list ${PATH_TO_ERASER2} ${PATH_TO_ERASER1} --eraser-before-lang-adapt