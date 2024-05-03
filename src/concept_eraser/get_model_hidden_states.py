import argparse

import torch.mps
from src.models import models
from src.utils import utils
import transformers
from tqdm import tqdm
import datasets
import torch
import os
import logging




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
TOP_LEVEL_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".."))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="biosbias",
        choices=[
            "biosbias"
        ],
        help='Dataset to use for training of concept erasure model.'
    )
    parser.add_argument(
        "--concept-label",
        type=str,
        default="gender",
        choices=[
            "gender",
            "profession"
        ],
        help='Label of concept for which an erasure model should be trained.'
    )
    parser.add_argument(
        "--dataset-language",
        type=str,
        default="en",
        choices=[
            "en"
        ],
        help='Language of dataset (used for initialization of adapter-based SwissBert).'
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help='Output directory to save hidden states and concept labels.'
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="dev",
        choices=[
            "train",
            "dev"
        ],
        help='Split of the dataset to use for training of concept erasure model.'
    )
    parser.add_argument(
        "--dataset-split-frac",
        type=float,
        default=0.2,
        help='Fraction of dataset split to use for training of concept erasure model in the interval (0, 1].'
    )
    parser.add_argument(
        "--intrasentence-model",
        type=str,
        default="BertForMLM",
        choices=[
            "BertForMLM",
            "SwissBertForMLM"
        ],
        help="Architecture of intrasentence model.",
    )
    parser.add_argument(
        "--pretrained-model-name",
        type=str,
        default="bert-base-uncased",
        choices=["bert-base-uncased", "ZurichNLP/swissbert-xlm-vocab"],
        help="Pretrained model name from which architecture weights are loaded.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to model checkpoint from which architecture weights are loaded.",
    )
    parser.add_argument(
        "--remove-lang-adapters-last-layer",
        default=False,
        action="store_true",
        help="Get hidden states before language adapter of last layer.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="ID of the experiment.",
    )
    args = parser.parse_args()
    return args


def get_dataset(dataset, split, frac):
    if dataset == "biosbias":
        perc = int(frac * 100)
        data = datasets.load_dataset("LabHC/bias_in_bios", split=f"{split}[:{perc}%]")
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")
    return data

def load_model_and_tokenizer(pretrained_model_name, intrasentence_model, ckpt_path, dataset_language):
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=True)
    model = getattr(models, intrasentence_model)(ckpt_path or pretrained_model_name, dataset_language)
    model.eval()
    model = model.to(DEVICE)
    return model, tokenizer

def remove_head(model):
    model.remove_head()
    return model

def remove_lang_adapters_in_layer(model):
    if model.model.name_or_path == "ZurichNLP/swissbert-xlm-vocab":
        model.remove_last_layer_adapter()
        model.remove_last_layer_layer_norm()
    else:
        logger.info("No language adapters to remove.")
    return model


def encode_dataset(tokenizer, dataset):
    def tokenization_no_padding(example):
            return tokenizer(example["hard_text"], padding = False, truncation = True,
                             return_token_type_ids = True, return_attention_mask = True,
                             add_special_tokens = True)
    dataset_enc = dataset.map(tokenization_no_padding, batched = True)
    return dataset_enc

def get_hidden_states(model, dataloader):
    cls_hidden_states = []
    avg_hidden_states = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            hidden_states = model(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
            # Compute average hidden states
            avg_hidden_states.append(hidden_states.squeeze(0).mean(dim=0))
            # Compute CLS token hidden state
            cls_hidden_states.append(hidden_states.squeeze(0)[0])
    
    avg_hidden_states = torch.cat(avg_hidden_states, dim=0)
    cls_hidden_states = torch.cat(cls_hidden_states, dim=0)

    hidden_size = model.model.config.hidden_size
    avg_hidden_states = torch.stack(avg_hidden_states.split(hidden_size))
    cls_hidden_states = torch.stack(cls_hidden_states.split(hidden_size))
    assert avg_hidden_states.size(1) == hidden_size
    assert cls_hidden_states.size(1) == hidden_size
    logger.info("Shape of hidden states: ")
    logger.info(f"Avg hidden states: {avg_hidden_states.shape}")
    logger.info(f"CLS hidden states: {cls_hidden_states.shape}")

    return avg_hidden_states, cls_hidden_states

def get_concept_label(data, concept_label):
    concept_labels = torch.tensor(data[concept_label])
    logger.info(f"Shape of concept labels: {concept_labels.shape}")
    return concept_labels

def main(args):
    logger.info("############# STARTED #############")
    logger.info(f"Args provided: {args}")
    logger.info(f"Using torch device: {DEVICE}")
    logger.info(f"Top level dir: {TOP_LEVEL_DIR}")

    if args.output_dir is None:
        output_dir = os.path.join(TOP_LEVEL_DIR, "data/concept_eraser/hidden_states")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    data = get_dataset(args.dataset, args.dataset_split, args.dataset_split_frac)

    concept_labels = get_concept_label(data, args.concept_label)

    model, tokenizer = load_model_and_tokenizer(args.pretrained_model_name, args.intrasentence_model, args.ckpt_path, args.dataset_language)
    dataset_enc = encode_dataset(tokenizer, data)
    data_collator = utils.CustomDataCollatorWithPadding(tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset_enc, batch_size = 1, shuffle = False, collate_fn = data_collator)

    model = remove_head(model)
    if args.remove_lang_adapters_last_layer:
        model = remove_lang_adapters_in_layer(model)


    avg_hidden_states, cls_hidden_states = get_hidden_states(model, dataloader)

    torch.save(concept_labels, f'{output_dir}/{args.dataset_split}_{args.concept_label}{("_" + args.experiment_id) if args.experiment_id else ""}.pt')
    torch.save(avg_hidden_states, f'{output_dir}/{args.dataset_split}_avg{("_" + args.experiment_id) if args.experiment_id else ""}.pt')
    torch.save(cls_hidden_states, f'{output_dir}/{args.dataset_split}_cls{("_" + args.experiment_id) if args.experiment_id else ""}.pt')

    logger.info("Concept labels and hidden states saved.")
    logger.info("############# FINISHED #############")



if __name__ == '__main__':
    args = parse_args()

    logging_dir = os.path.join(TOP_LEVEL_DIR, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(logging_dir, "get_model_hidden_states.log"),
                    format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args)