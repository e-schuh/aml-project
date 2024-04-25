from collections import defaultdict
import torch
from tqdm import tqdm
import logging

from src.dataloader import dataloader
from src.utils import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger = logging.getLogger(__name__)

class IntersentenceInferenceRunner:
    def __init__(self, model, tokenizer, data_path, pretrained_model_name, batch_size, tiny_eval_frac, softmax_temperature):
        pass

    def run(self):
        pass