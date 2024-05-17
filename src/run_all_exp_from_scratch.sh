#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"


echo "--Prepare Concept Eraser Experiments--"
./concept_eraser/run.sh


echo "--Prepare Refine LM Experiments--"
./refine_lm/run.sh


echo "--Run Inference of Different Experiments--"

echo "Running Inference for Baseline Model..."
./run_inference_baseline.sh

echo "Running Inference for different Softmax Temperatures..."
./run_inference_temperature_experiments.sh

echo "Running Inference for Concept Eraser Models..."
./run_inference_concept_eraser_experiments.sh

echo "Running Inference for Refine LM Models..."
./run_inference_refine_lm_experiments.sh


echo "--Run Evaluation of Different Experiments--"

echo "Running Evaluation for Baseline Model..."
./run_evaluation_baseline.sh

echo "Running Evaluation for different Softmax Temperatures..."
./run_evaluation_temperature_experiments.sh

echo "Running Evaluation for Concept Eraser Models..."
./run_evaluation_concept_eraser_experiments.sh

echo "Running Evaluation for Refine LM Models..."
./run_evaluation_refine_lm_experiments.sh