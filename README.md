## Installation
1. Clone the repository: `git clone https://github.com/e-schuh/aml-project.git`
2. Navigate into top-level directory and install the requirements: `cd AML && pip install -r requirements.txt`
3. Install the project as editable package: `pip install -e .`

## Reproduction of experiments
### Reproducing all experiments from scratch
All experiments can be reproduced by invoking a single command from the top-level AML directory. It will run the following steps in order for each experiment setup: data preprocessing / experiment preparations, training, inference, and evaluation. __Note that on a CPU device, this might take up to 48h to run.__
```
sh src/run_all_exp_from_scratch.sh
```
The evaluation results of the experiments will be saved in the 'evaluation_output' directories.

### Reproducing an individual experiment from scratch
Furthermore, we provide simple shell scripts to run the individual experiments also in isolation. For example, to reproduce the Concept Eraser experiments, run the following commands from the top-level AML directory:
```
#This runs the experiment preparations and trains different the concept erasure models
sh src/concept_eraser/run.sh

#This runs the inference of SwissBERT with the different concept erasure models applied to it
sh src/run_inference_concept_eraser_experiments.sh 

#This evaluates the predictions of the concept erasure augmented SwissBERT model (results are saved in the 'evaluation_output' directory)
sh src/run_evaluation_concept_eraser_experiments.sh
```

## Brief overview of the underlying code structure
The following sections provide a brief overview of the code structure and the main components of our AML project. For a more detailed description of the exact functionality of each component, the `--help` flag can be run for each python module (e.g., `python src/inference.py --help`).

### Run inference
First, run inference/predictions for the model under consideration (following commands are from the top-level AML directory):

- Bert / Intrasentence: `python src/inference.py`

- SwissBert / Intrasentence / DE: `python src/inference.py  --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path "./data/df_intrasentence_de.pkl"`

- SwissBert / Intrasentence / EN: `python src/inference.py  --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path "./data/df_intrasentence_en.pkl"`

The inference output will be saved in the 'inference_output' directory by default.

Note: `inference.py` accepts optional arguments such as batch size (`--batch-size`), subsample of the entire data (`--tiny-eval-frac`), or different data, model, and output paths. Also note the default values. For more information, run `python src/inference.py --help`.

If a GPU is available, it will automatically be used. If on an Apple Silicon device, MPS will automatically be used. Else, CPU will be used.

### Evaluate predictions
To evaluate the generated predictions from above, run the following command from the top-level AML directory:

- Bert / Intrasentence: `python src/evaluation.py --intrasentence-gold-file-path "./data/df_intrasentence_en.pkl" --inference-output-file "./inference_output/combined_results_BertForMLM_en.json" --skip-intersentence`

- SwissBert / Intrasentence / DE: `python src/evaluation.py --intrasentence-gold-file-path "./data/df_intrasentence_de.pkl" --inference-output-file "./inference_output/combined_results_SwissBertForMLM_de.json" --skip-intersentence`

- SwissBert / Intrasentence / EN: `python src/evaluation.py --intrasentence-gold-file-path "./data/df_intrasentence_en.pkl" --inference-output-file "./inference_output/combined_results_SwissBertForMLM_en.json" --skip-intersentence`

The evaluation output (i.e., the different scores) will be saved in the 'evaluation_output' directory by default.

Note: Adjust the paths provided to `evaluation.py` as needed. The gold file path (`--intrasentence-gold-file-path`) points to the ground truth data frame (choose the appropriate language that matches the inference file); the inference output path (`--inference-output-file`) to the file created by running inference from the above step. For more information, run `python src/evaluation.py --help`.

### Concept erasure
_Note: If on an Apple silicon device (M1, M2, M3,...), one needs to run `export PYTORCH_ENABLE_MPS_FALLBACK=1` in the command line before running the following commands for concept erasure (because certain operations of concept-erasure are not yet natively supported on Apple silicon devices)._

To train a concept erasure model ([Belrose et al. (2023)](https://arxiv.org/pdf/2306.03819)) and apply it to the SwissBert model, run the following commands from the top-level AML directory:

1. Pre-compute and save the SwissBert hidden states before the language modelling head. Specify the concept to be removed via the `--concept-label` arugment (currently, the concepts `"gender"` and `"profession"` are supported; `"gender"` being the default value): `python src/concept_eraser/get_model_hidden_states.py --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab"`

2. Train the concept erasure model using the pre-computed hidden states. Ensure that the arguments `--path-to-hidden-states` and `--path-to-concept-labels` match the saved .pt files from the previous step (by default, the `"gender"` concept is used): `python src/concept_eraser/train_concept_erasure_model.py`

3. Run inference as described above for SwissBert but with the concept erasure models (indicated by the `--eraser-path-list` argument pointing to the .pkl files of the trained concept erasure models from previous step; if no specific path is provided with the flag, the default save path of the previous step will be used). If multiple erasers are specified in the `--eraser-path-list` argument, the erasers are applied sequentially in the same order as specified (e.g., `--eraser-path-list "data/concept_eraser/eraser_models/eraser_gender.pkl" "data/concept_eraser/eraser_models/eraser_profession.pkl"` will first apply the gender erasure model, followed by the profession erasure model): `python src/inference.py --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path "./data/df_intrasentence_de.pkl" --eraser-path-list <path-to-eraser-model-1> <path-to-eraser-model-2>`

4. Evaluate the predictions as described above.


### Refine-LM
To use the refine-lm approach model ([Qureshi et al. (2024)](https://inria.hal.science/hal-04426115/file/NAACL_2023_Refine_LM%20%281%29.pdf)) and apply it to the SwissBert model, run the following commands from the top-level AML directory:

1. Generate underspecified examples: `sh src/refine_lm/generate_us_examples.sh`

2. Run preprocessing and generate .pkl files: `sh src/refine_lm/run_preprocessing.sh`

3. Train model: `sh src/refine_lm/train_bert.sh`

4. Run inference as described above for SwissBert but with the saved model checkpoints from the previous training step (indicated by the `--ckpt-path` argument pointing to the .pth file of the trained model from previous step; e.g., `--ckpt-path "./data/refine_lm/saved_models/swissbert_o_gender_tk8_ep1.pth"`):`python src/inference.py  --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --ckpt-path <path-to-model-checkpoint> --intrasentence-data-path "./data/df_intrasentence_de.pkl" --experiment-id refine-lm`

5. Evaluate the predictions as described above.
