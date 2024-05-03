from collections import defaultdict
import torch
from tqdm import tqdm
import logging

from src.dataloader import dataloader
from src.utils import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger = logging.getLogger(__name__)

class IntrasentenceInferenceRunner:
    def __init__(self, model, tokenizer, data_path, pretrained_model_name, batch_size, tiny_eval_frac, softmax_temperature):
        self._model = model
        self._tokenizer = tokenizer
        self._data_path = data_path
        self._pretrained_model_name = pretrained_model_name
        self._batch_size = batch_size
        self._tiny_eval_frac = tiny_eval_frac
        self._softmax_temperature = softmax_temperature
        self._mask_token = self._tokenizer.mask_token
        self._mask_token_id = self._tokenizer.mask_token_id

        self._model.eval()

    def run(self):
        # Modified based on Öztürk et al. (2023) https://arxiv.org/abs/2307.07331
        """Score intrasentence examples using likelihood scoring as proposed by Nadeem et al.

        Likelihood scoring computes the masked word probability of the stereotypical, anti-stereotypical,
        and unrelated associations for a given example. If a candidate consists of multiple subtokens,
        the score is computed by averaging the log probability of each subtoken.
        """
        model = self._model.to(DEVICE)

        self._do_dynamic_padding = True
        self._do_shuffle = False

        dataset = dataloader.IntrasentenceDataset(
            self._data_path,
            self._tokenizer,
            self._pretrained_model_name,
            self._tiny_eval_frac,
            self._do_dynamic_padding
        )

        loader = dataset.get_dataloader(self._batch_size, self._do_shuffle)
        logger.info(f"Dataloader ready.")

        word_probabilities = defaultdict(list)

        # Calculate the logits for each prediction.
        logger.info("Word probability calculations starts for intrasentence...")
        for batch in tqdm(loader, total=len(loader)):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            masked_tokens = batch["masked_tokens"]
            sentence_id = batch["sentence_id"]

            # Move to DEVICE
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            masked_tokens = masked_tokens.to(DEVICE)

            # Get index of the masked token
            mask_idxs = (input_ids == self._mask_token_id)

            # Get the probabilities for every token in the sentence
            with torch.no_grad():
                logits = model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = logits / self._softmax_temperature
                output = logits.softmax(dim=-1)
            
            # Extract the probabilities for only the masked token in the sentence
            output = output[mask_idxs]
            output = output.index_select(1, masked_tokens.squeeze(-1)).diag()
            
            # Append the probabilities of masked token for each sentence_id in the batch
            # to the word_probabilities dictionary
            for idx, probs in enumerate(output):
                word_probabilities[sentence_id[idx]].append(probs.item())

        # Reconcile the probabilities into sentences.
        logger.info("Word probability calculations finished. Aggregating scores of subtokens...")
        sentence_probabilities = []
        for k, v in word_probabilities.items():
            score = utils.aggregate_scores(v)
            pred = {"id": k, "score": score}
            sentence_probabilities.append(pred)
        
        logger.info("Intrasentence inference finished.")
        return sentence_probabilities
