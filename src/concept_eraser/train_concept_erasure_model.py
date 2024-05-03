import torch
import argparse
import os
import logging

from concept_erasure import LeaceEraser
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
TOP_LEVEL_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), ".."))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-hidden-states",
        type=str,
        default=os.path.join(TOP_LEVEL_DIR, "data/concept_eraser/hidden_states/dev_cls.pt"),
        help='Path to saved hidden states.'
    )
    parser.add_argument(
        "--path-to-concept-labels",
        type=str,
        default=os.path.join(TOP_LEVEL_DIR, "data/concept_eraser/hidden_states/dev_gender.pt"),
        help='Path to saved concept labels.'
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(TOP_LEVEL_DIR, "data/concept_eraser/eraser_models"),
        help='Output directory to save trained concept erasure model.'
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="ID of the experiment.",
    )
    args = parser.parse_args()
    return args


def load_data(path):
    data = torch.load(path)
    return data

def main(args):
    logger.info("############# STARTED #############")
    logger.info(f"Args provided: {args}")
    logger.info(f"Using torch device: {DEVICE}")
    logger.info(f"Top level dir: {TOP_LEVEL_DIR}")

    os.makedirs(args.output_dir, exist_ok=True)

    hidden_states = load_data(args.path_to_hidden_states)
    concept_labels = load_data(args.path_to_concept_labels)

    hidden_states = hidden_states.to(DEVICE)
    concept_labels = concept_labels.to(DEVICE)

    eraser = LeaceEraser.fit(hidden_states, concept_labels)
    logger.info(f"Device of eraser: {eraser.P.device}")
    with open(f'{args.output_dir}/eraser{("_" + args.experiment_id) if args.experiment_id else ""}.pkl', "wb") as f:
        pickle.dump(eraser, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info("Concept eraser saved.")
    logger.info("############# FINISHED #############")


if __name__ == "__main__":
    args = parse_args()

    logging_dir = os.path.join(TOP_LEVEL_DIR, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(logging_dir, "train_concept_erasure_model.log"),
                    format='%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(args)