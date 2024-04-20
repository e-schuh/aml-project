import transformers
import torch
import logging

logger = logging.getLogger(__name__)

class BertForMLM:
    def __new__(cls, pretrained_model_name, *args, **kwargs):
        return transformers.BertForMaskedLM.from_pretrained(pretrained_model_name)
    
class BertForNSP:
    def __new__(cls, pretrained_model_name, *args, **kwargs):
        return transformers.BertForNextSentencePrediction.from_pretrained(pretrained_model_name)
    
class SwissBertForMLM:
    def __new__(cls, pretrained_model_name, language, *args, **kwargs):
        languages = ["de_CH", "fr_CH", "it_CH", "rm_CH", "en_XX"]
        swissBert = cls._get_new_swissBert(pretrained_model_name, languages)
        if language == "de":
            swissBert.set_default_language("de_CH")
            logger.info("SwissBERT language set to de_CH")
        elif language == "en":
            swissBert.set_default_language("en_XX")
            logger.info("SwissBERT language set to en_XX")
        else:
            raise ValueError(f"Language {language} not supported by SwissBERT. Supported languages: de, en")
        return swissBert

    @classmethod
    def _get_new_swissBert(cls, pretrained_model_name, languages):
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