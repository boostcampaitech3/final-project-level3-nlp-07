import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel


class SktKoGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        MODEL_NAME = "skt/kogpt2-base-v2"
        self.max_length_token = 128
        special_tokens = []

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME,
                bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                pad_token='<pad>', mask_token='<mask>', special_tokens=special_tokens)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    
    def tokenize(self, sent):
        return self.tokenizer.tokenize(sent) # tokenized string (with prefix '_')
    
    def forward(self, x) -> torch.Tensor:
        # 별점/메뉴/고객리뷰 입력 => 사장답글

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


def get_model(model_path: str = None) -> nn.Module:
    model = SktKoGPT2()#.to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path)) # , map_location=device
    return model

def predict_from_model(model: nn.Module, input_string: str) -> str:
    output_string = model.forward(input_string) # str을 cuda로 보내는 방법?
    return output_string
