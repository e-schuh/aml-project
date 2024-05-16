import torch
from torch import nn
from transformers import AutoTokenizer 
from src.models import models

class CustomBERTModel(nn.Module):
	def __init__(self, k, batch_size, intrasentence_model = "SwissBertForMLM", pretrained_model_name = "ZurichNLP/swissbert-xlm-vocab", language = "en"):
		super(CustomBERTModel, self).__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, return_token_type_ids=False, use_fast=True)
		self.bert = getattr(models, intrasentence_model)(pretrained_model_name, language=language)
		self.topk = k
		self.batch_size = batch_size
		self.out = nn.Linear(self.topk, self.topk)
		nn.init.ones_(self.out.weight)

	def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
		output = self.bert(input_ids)
		inputs = torch.zeros_like(output)  

		for i in range(len(output)): # number of sentence 
			masked_index = (input_ids[i] == self.tokenizer.mask_token_id)
			j = masked_index
			logits = output[i, j, :] #[sentence, word, probable token]

			values, indices = logits.topk(self.topk)

			output_values  = self.out(values)

			layer_output = output_values.softmax(dim=-1)

			#Filling in the topk logits in the dictionary with the actual values
			for k in range(indices.shape[-1]):
				inputs[i, j, indices[:,k]] = layer_output[:,k]
		return inputs