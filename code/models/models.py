import transformers

class BertForMLM:
    def __new__(cls, pretrained_model_name):
        return transformers.BertForMaskedLM.from_pretrained(pretrained_model_name)
    
class BertForNSP:
    def __new__(cls, pretrained_model_name):
        return transformers.BertForNextSentencePrediction.from_pretrained(pretrained_model_name)
    
class SwissBertForMLM:
    def __new__(cls, pretrained_model_name):
        raise NotImplementedError

class SwissBertForNSP:
    def __new__(cls, pretrained_model_name):
        raise NotImplementedError