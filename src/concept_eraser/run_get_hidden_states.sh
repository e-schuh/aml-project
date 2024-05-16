#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

## CONCEPT=gender; after language adapters ##
CONCEPT="gender"
echo ">> Computing hidden states after language adapter for concept "${CONCEPT}
python3 -m get_model_hidden_states --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --concept-label ${CONCEPT}


## CONCEPT=gender; before language adapters ##
CONCEPT="gender"
echo ">> Computing hidden states before language adapter for concept "${CONCEPT}
python3 -m get_model_hidden_states --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --concept-label ${CONCEPT} --remove-lang-adapters-last-layer


## CONCEPT=profession; after language adapters ##
CONCEPT="profession"
echo ">> Computing hidden states after language adapter for concept "${CONCEPT}
python3 -m get_model_hidden_states --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --concept-label ${CONCEPT}


## CONCEPT=profession; before language adapters ##
CONCEPT="profession"
echo ">> Computing hidden states before language adapter for concept "${CONCEPT}
python3 -m get_model_hidden_states --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --concept-label ${CONCEPT} --remove-lang-adapters-last-layer