## Installation
1. Clone the repository: `git clone https://github.com/katjahager/AML.git`
2. Navigate into top-level directory and install the requirements: `cd AML && pip install -r requirements.txt`
3. Install the project as editable package: `pip install -e .`

## Run inference
First, run inference/predictions for the model under consideration (following commands are from the top-level AML directory):

1. Bert / Intrasentence: `python src/inference.py`

2. SwissBert / Intrasentence / DE: `python src/inference.py  --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path "./data/df_intrasentence_de.pkl"`

3. SwissBert / Intrasentence / EN: `python src/inference.py  --intrasentence-model "SwissBertForMLM" --pretrained-model-name "ZurichNLP/swissbert-xlm-vocab" --intrasentence-data-path "./data/df_intrasentence_en.pkl"`

The inference output will be saved in the 'inference_output' directory by default.

Note: `inference.py` accepts optional arguments such as batch size (`--batch-size`), subsample of the entire data (`--tiny-eval-frac`), or different data, model, and output paths. Also note the default values. For more information, run `python src/inference.py --help`.

If a GPU is available, it will automatically be used. If on an Apple Silicon device, MPS will automatically be used. Else, CPU will be used.

## Evaluate predictions
To evaluate the generated predictions from above, run the following command from the top-level AML directory:

1. Bert / Intrasentence: `python src/evaluation.py --intrasentence-gold-file-path "./data/df_intrasentence_en.pkl" --inference-output-file "./inference_output/combined_results_BertForMLM_en.json" --skip-intersentence`

2. SwissBert / Intrasentence / DE: `python src/evaluation.py --intrasentence-gold-file-path "./data/df_intrasentence_de.pkl" --inference-output-file "./inference_output/combined_results_SwissBertForMLM_de.json" --skip-intersentence`

3. SwissBert / Intrasentence / EN: `python src/evaluation.py --intrasentence-gold-file-path "./data/df_intrasentence_en.pkl" --inference-output-file "./inference_output/combined_results_SwissBertForMLM_en.json" --skip-intersentence`

The evaluation output (i.e., the different scores) will be saved in the 'evaluation_output' directory by default.

Note: Adjust the paths provided to `evaluation.py` as needed. The gold file path (`--intrasentence-gold-file-path`) points to the ground truth data frame (choose the appropriate language that matches the inference file); the inference output path (`--inference-output-file`) to the file created by running inference from the above step. For more information, run `python src/evaluation.py --help`.