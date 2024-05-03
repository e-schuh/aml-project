import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase, BatchEncoding
import logging
import torch
import functools
import argparse

logger = logging.getLogger(__name__)


def aggregate_scores(scores):
    # result = np.sum([np.log2(i) for i in v]) + np.log2(len(v))
    # result = np.mean(scores)

    # Scores are list of probabilities for all subtokens that make up a candidate word (can be 1, if the candidate word is in the vocab, or more than 1 if the candidate word is OOV)
    # Following Kauf et al. (2023) https://aclanthology.org/2023.acl-short.80.pdf, we represent the probability of the candidate word as the sum of the log probabilities of its subtokens
    result = np.sum([np.log(i) for i in scores])

    return result


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))



class CustomIdentity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inp, lang, *args, **kwargs):
        return args[0]


@dataclass
class CustomDataCollatorWithPadding:
    # Modified based on https://github.com/huggingface/transformers/issues/28066#issuecomment-1884522968
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer : PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        padding_features = [{key : val for key, val in row.items() if key in ['input_ids','attention_mask', 'token_type_ids']} for row in features]
        
        batch = self.tokenizer.pad(
            padding_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        batch = BatchEncoding(batch, tensor_type=self.return_tensors)
        
        for row in features:
            for key, value in row.items():
                if key in ['input_ids','attention_mask','token_type_ids']:
                    continue
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)
        
        try:
            batch['masked_tokens'] = torch.stack(batch['masked_tokens'])
        except:
            pass
        return batch

class customAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if option_string:
            if values:
                setattr(namespace, self.dest, values)
            else:
                setattr(namespace, self.dest, self.const)
        else:
            setattr(namespace, self.dest, None)