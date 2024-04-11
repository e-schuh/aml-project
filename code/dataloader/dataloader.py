import torch
import pandas as pd
import logging
from tqdm import tqdm
from datasets import Dataset
from utils import utils


logger = logging.getLogger(__name__)

class IntrasentenceDataset(torch.utils.data.Dataset):
    # Modified based on Öztürk et al. (2023) https://arxiv.org/abs/2307.07331
    def __init__(self, data_path, tokenizer, pretrained_model_name, tiny_eval_frac, do_dynamic_padding):
        self._tokenizer = tokenizer
        self._pretrained_model_name = pretrained_model_name
        self._tiny_eval_frac = tiny_eval_frac
        self._do_dynamic_padding = do_dynamic_padding

        self._df = pd.read_pickle(data_path)
        if self._tiny_eval_frac:
            self._df = self._df.sample(frac=self._tiny_eval_frac, random_state=42)
            logger.info(f"Tiny data frame loaded. Number of data records: {len(self._df)} ({self._tiny_eval_frac*100}% of entire data)")

        self._sentences = self._create_sentences_from_df(self._df)
        self._dataset = self._create_dataset_from_sentence_list(self._sentences)

        if self._do_dynamic_padding:
            self._dataset = self._encode_dataset_no_pad(self._dataset, self._tokenizer)
        else:
            self._dataset = self._encode_dataset_pad_max_length(self._dataset, self._tokenizer)
        
        self._dataset.set_format(type = "torch", columns=["input_ids", "attention_mask", "token_type_ids", "masked_tokens"], output_all_columns=True)
        
        logger.info("Dataset is encoded and ready.")
        logger.info(f"Example record of final dataset: {self._dataset[0]}")



    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def get_dataloader(self, batch_size, shuffle):
        if self._do_dynamic_padding:
            data_collator = utils.CustomDataCollatorWithPadding(self._tokenizer)
            dataloader = torch.utils.data.DataLoader(self._dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = data_collator)
        else:
            dataloader = torch.utils.data.DataLoader(self._dataset, batch_size = batch_size, shuffle = shuffle)
        return dataloader

    def _create_sentences_from_df(self, df):
        final_sentence_list = []
        logger.info("Creating sentences from dataset...")
        for index, example in tqdm(df.iterrows(), total=df.shape[0]):
            for candidate_id in range(1, 4):
                self._add_discriminative_intrasentence_candidates(example, candidate_id, final_sentence_list)
        logger.info(f"Finished. Total number of sentences created: {len(final_sentence_list)}")
        logger.info(f"Example sentence: {final_sentence_list[0]}")
        return final_sentence_list
    
    def _add_discriminative_intrasentence_candidates(self, example, candidate_id, final_sentence_list):
        insertion_tokens = self._tokenizer.encode(example[f"c{candidate_id}_word"], add_special_tokens=False)
        for idx in range(len(insertion_tokens)):
            insertion = self._tokenizer.decode(insertion_tokens[:idx])
            new_sentence = example["context"].replace("BLANK", f"{insertion}{self._tokenizer.mask_token}")
            final_sentence_list.append({"sentence": new_sentence, "candidate_id": example[f"c{candidate_id}_id"],
                                   "masked_token": torch.tensor(insertion_tokens[idx])})
            
    def _create_dataset_from_sentence_list(self, sentence_list):
        logger.info("Creating dataset from sentence list...")
        dataset = dict.fromkeys(["sentence", "masked_tokens", "sentence_id"])
        for key in dataset.keys():
            dataset[key] = []
        for sentence_dict in sentence_list:
            dataset["sentence"].append(sentence_dict["sentence"])
            dataset["masked_tokens"].append(sentence_dict["masked_token"].unsqueeze(-1))
            dataset["sentence_id"].append(sentence_dict["candidate_id"])
        
        dataset = Dataset.from_dict(dataset)
        return dataset
    
    def _encode_dataset_pad_max_length(self, dataset, tokenizer):
        logger.info("Encoding dataset with padding to max length...")
        def tokenization_pad_max_length(example):
            return tokenizer(example["sentence"], padding = 'max_length', truncation = True,
                             return_token_type_ids = True, return_attention_mask = True)
        dataset_enc = dataset.map(tokenization_pad_max_length, batched = True)
        return dataset_enc

    def _encode_dataset_no_pad(self, dataset, tokenizer):
        logger.info("Encoding dataset without padding (padding is done later dynamically at batch level)...")
        def tokenization_no_padding(example):
            return tokenizer(example["sentence"], padding = False, truncation = True,
                             return_token_type_ids = True, return_attention_mask = True)
        dataset_enc = dataset.map(tokenization_no_padding, batched = True)
        return dataset_enc
        




class IntersentenceDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def get_dataloader(self, batch_size, shuffle):
        pass