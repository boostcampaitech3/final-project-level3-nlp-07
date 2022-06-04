import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel, BartForConditionalGeneration

special_tokens = ['#@상호명#', '#@고객이름#', '#@위치#', '#@기관#', '#@전화번호#']

class KoBART(nn.Module):
    def __init__(self, model_path: str = None):
        super().__init__()
        MODEL_NAME = "gogamza/kobart-base-v2"

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME,
                bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                pad_token='<pad>', mask_token='<mask>', special_tokens=special_tokens)
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.model = BartForConditionalGeneration.from_pretrained(model_path if model_path else MODEL_NAME)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, sent) -> torch.Tensor:
        sent = sent.replace('\n', '')

        input_ids = self.tokenizer.encode(sent, return_tensors='pt')
        output = self.model.generate(input_ids, 
                                    eos_token_id=self.tokenizer.eos_token_id, 
                                    max_length=128, 
                                    num_beams=5, 
                                    no_repeat_ngram_size=2, 
                                    early_stopping=True,
                                    use_cache=True)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output


class SktKoGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        MODEL_NAME = "skt/kogpt2-base-v2"
        self.max_length_token = 128
        special_tokens = ['#@상호명#', '#@고객이름#', '#@위치#', '#@기관#', '#@전화번호#']

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME,
                bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                pad_token='<pad>', mask_token='<mask>', special_tokens=special_tokens)
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def tokenize(self, sent):
        return self.tokenizer.tokenize(sent) # tokenized string (with prefix '_')
    
    def forward(self, x) -> torch.Tensor:
        input_ids = self.tokenizer.encode(x, return_tensors='pt') # encoded tokens
        gen_ids = self.model.generate(input_ids,
                                        max_length=self.max_length_token,
                                        repetition_penalty=2.0,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        bos_token_id=self.tokenizer.bos_token_id,
                                        use_cache=True)
        generated = self.tokenizer.decode(gen_ids[0])
        return generated


def get_model(model_type: str = 'KoBART', model_path: str = None) -> nn.Module:
    if model_type == 'KoBART':
        model = KoBART(model_path=model_path)
    
    elif model_type == 'SktKoGPT2':
        model = SktKoGPT2()
        if model_path:
            model.load_state_dict(torch.load(model_path))
    return model

def predict_from_model(model: nn.Module, input_string: str) -> str:
    output_string = model.forward(input_string)
    return output_string
