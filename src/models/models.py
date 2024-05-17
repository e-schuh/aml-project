import transformers
import torch
import logging
import copy
import pickle
from types import MethodType
from transformers import logging as hf_logging


from src.utils import utils
from src.refine_lm import model_BERT

logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()
SWISSBERT_LANGUAGES = ["de_CH", "fr_CH", "it_CH", "rm_CH", "en_XX"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class BertForMLM(torch.nn.Module):
    def __init__(self, pretrained_model_name, language, eraser_paths=None, *args, **kwargs):
        super().__init__()
        self.model = self._get_new_Bert(pretrained_model_name).to(DEVICE)
        self.erasers = _load_erasers(eraser_paths)
        
        if self.erasers is not None:
            # Copy the lm head to later re-apply on top of concept-erased hidden states
            self.lm_head = self._get_copy_lm_head()

            # Deactivate lm head to get hidden states before lm head
            self._remove_head()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if isinstance(self.model, model_BERT.CustomBERTModel):
            probs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
                )
            return probs
        if self.erasers is None:
            logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
                )[0]
            return logits
        
        else:
            # Get hidden states before lm head
            hidden_states = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
                )[0]

            # Erase concepts from hidden states
            for eraser in self.erasers:
                hidden_states = eraser(hidden_states)
            out = hidden_states

            # Re-apply lm head
            logits = self.lm_head(out)

            return logits
    
    def _get_copy_lm_head(self):
        return copy.deepcopy(utils.rgetattr(self.model, "cls"))
    
    def _remove_head(self):
        utils.rsetattr(self.model, "cls", torch.nn.Identity())


    def _get_new_Bert(self, pretrained_model_name):
        bert = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model_name)
        return bert
    
class BertForNSP:
    def __new__(cls, pretrained_model_name, *args, **kwargs):
        return transformers.BertForNextSentencePrediction.from_pretrained(pretrained_model_name)
    
class SwissBertForMLM(torch.nn.Module):
    def __init__(self, pretrained_model_name, language, eraser_paths=None, is_eraser_before_lang_adapt=False, *args, **kwargs):
        super().__init__()
        if pretrained_model_name.endswith((".pt", ".pth")):
            self.model = self._get_saved_model(pretrained_model_name, DEVICE)
        else:
            self.model = self._get_new_swissBert(pretrained_model_name, SWISSBERT_LANGUAGES).to(DEVICE)
        
        self.erasers = _load_erasers(eraser_paths)
        self._is_eraser_before_lang_adapt = is_eraser_before_lang_adapt

        if language == "de":
            if isinstance(self.model, model_BERT.CustomBERTModel):
                self.model.bert.model.set_default_language("de_CH")
            else:
                self.model.set_default_language("de_CH")
            logger.info("SwissBERT language set to de_CH")
        elif language == "en":
            if isinstance(self.model, model_BERT.CustomBERTModel):
                self.model.bert.model.set_default_language("en_XX")
            else:
                self.model.set_default_language("en_XX")
            logger.info("SwissBERT language set to en_XX")
        else:
            raise ValueError(f"Language {language} not supported by SwissBERT. Supported languages: de, en")
        
        if self.erasers is not None:
            # Copy the original language adapter, layer norm and lm head to later re-apply on top of
            # concept-erased hidden states
            self.lm_head = self._get_copy_lm_head()
            if self._is_eraser_before_lang_adapt:
                self.adapter = self._get_copy_last_layer_adapter()
                self.layer_norm = self._get_copy_last_layer_layer_norm()

            # Deactivate language adapter, layer norm and lm head to get hidden states before language adapter, layer norm and lm head
            self._remove_head()
            if self._is_eraser_before_lang_adapt:
                self._remove_last_layer_adapter()
                self._remove_last_layer_layer_norm()
    
    def _get_copy_last_layer_adapter(self):
        return copy.deepcopy(utils.rgetattr(self.model, "roberta.encoder.layer.11.output.lang_adapter"))
    
    def _get_copy_last_layer_layer_norm(self):
        return copy.deepcopy(utils.rgetattr(self.model, "roberta.encoder.layer.11.output.LayerNorm"))
    
    def _get_copy_lm_head(self):
        return copy.deepcopy(utils.rgetattr(self.model, "lm_head"))
    
    def _remove_head(self):
        utils.rsetattr(self.model, "lm_head", torch.nn.Identity())

    def _remove_last_layer_adapter(self):
        self.model.roberta.encoder.layer[11].output.lang_adapter = MethodType(utils.CustomIdentity(), self.model.roberta.encoder.layer[11].output)

    def _remove_last_layer_layer_norm(self):
        utils.rsetattr(self.model, "roberta.encoder.layer.11.output.LayerNorm", torch.nn.Identity())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if isinstance(self.model, model_BERT.CustomBERTModel):
            probs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
                )
            return probs
        if self.erasers is None:
            logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
                )[0]
            return logits
        
        else:
            # Get hidden states before language adapter, layer norm and lm head
            hidden_states = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
                )[0]

            # Erase concepts from hidden states
            for eraser in self.erasers:
                hidden_states = eraser(hidden_states)
            out = hidden_states

            if self._is_eraser_before_lang_adapt:
                batch_size = out.size()[0]
                lang_ids = 0 * torch.ones(batch_size, device=DEVICE)
                
                # Re-apply language adapter on erased hidden states
                out = self.adapter(lang_ids, out)

                # Re-apply layer norm
                out = self.layer_norm(out)

            # Re-apply lm head
            logits = self.lm_head(out)

            return logits

    def _get_saved_model(self, ckpt_path, device=DEVICE):
        cfg = torch.load(ckpt_path, map_location=device)
        state_dict = cfg["state_dict"]
        topk = cfg["topk"]
        batch_size = cfg["batch_size"]
        model = model_BERT.CustomBERTModel(topk, batch_size, "SwissBertForMLM", "ZurichNLP/swissbert-xlm-vocab")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def _get_new_swissBert(self, pretrained_model_name, languages):
        # Load SwissBERT with XLM Vocab ("Variant 1" in paper) with config file
        # Note: Params of config file which are not in pre-trained "ZurichNLP/swissbert-xlm-vocab" model (i.e., the en_XX language adapter)
        # get randomly initialized
        swissBert_config = transformers.AutoConfig.from_pretrained(pretrained_model_name, languages = languages)

        swissBert = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model_name, config = swissBert_config)

        # Load X-MOD model and get its state_dict
        xmod = transformers.AutoModel.from_pretrained("facebook/xmod-base")
        xmod_sd = xmod.state_dict()

        # Remove all params except for the en_XX language adapter
        layers_to_remove = []
        for key in xmod_sd:
            if "en_XX" not in key:
                layers_to_remove.append(key)

        for key in layers_to_remove:
            del xmod_sd[key]

        # Load the en_XX language adapter params from X-MOD into SwissBERT (need to prepend 'roberta.' due to different layer namings in SwissBERT)
        swissBert.load_state_dict({f"roberta.{k}":v for k,v in xmod_sd.items()}, strict=False)

        # Check that all en_XX language adapter params are equal to X-MOD and all other params are equal to SwissBERT
        swissBert_new = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model_name)
        xmod_new = transformers.AutoModel.from_pretrained("facebook/xmod-base")

        swissBERT_sd = swissBert.state_dict()
        swissBERT_new_sd = swissBert_new.state_dict()
        xMod_new_sd = xmod_new.state_dict()

        mismatch = 0
        for key in swissBERT_sd:
            if "en_XX" in key:
                if not torch.equal(swissBERT_sd[key], xMod_new_sd[key.partition("roberta.")[-1]]):
                    logger.info(f"SwissBert initialization: Params of layer {key} do not match")
                    mismatch+=1
            else:
                if not torch.equal(swissBERT_sd[key], swissBERT_new_sd[key]):
                    logger.info(f"SwissBert initialization: Params of layer {key} do not match")
                    mismatch+=1
        if not mismatch:
            logger.info("SwissBert initialization: All params match")
        else:
            raise ValueError("SwissBert initialization: Not all params match")
        
        return swissBert

class SwissBertForNSP:
    def __new__(cls, pretrained_model_name):
        raise NotImplementedError
    
def _load_erasers(eraser_paths):
    if eraser_paths is None:
        return None
    
    erasers = []
    for eraser_path in eraser_paths:
        with open(eraser_path, 'rb') as path:
            eraser = pickle.load(path)
        erasers.append(eraser)
    return erasers