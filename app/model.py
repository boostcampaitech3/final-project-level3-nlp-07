import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel, BartForConditionalGeneration
import streamlit as st
from app.postprocess import postprocess

special_tokens = ['#@상호명#', '#@고객이름#', '#@위치#', '#@기관#', '#@전화번호#']


@st.cache
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

    def forward(self, input_str, store_name, customer_name) -> torch.Tensor:
        input_str = input_str.replace('\n', '')

        input_ids = self.tokenizer.encode(input_str, return_tensors='pt')
        output = self.model.generate(input_ids, 
                                    eos_token_id=self.tokenizer.eos_token_id, 
                                    max_length=128, 
                                    num_beams=5, 
                                    no_repeat_ngram_size=2, 
                                    early_stopping=True,
                                    use_cache=True)

        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        processed_output = postprocess(output, store_name=store_name, customer_name=customer_name)
        return processed_output


class KoGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        MODEL_NAME = "skt/kogpt2-base-v2"
        self.max_length_token = 128

        self.REVIEW = '<unused1>'
        self.PTUNING = '<unused2>'
        self.BOS = '<bos>'
        self.EOS = '<eos>'

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_str, store_name, customer_name) -> torch.Tensor:
        input_str = input_str.replace('\n', '')
        
        input_ids = self.tokenizer.encode(self.BOS) + self.tokenizer.encode(input_str) + self.tokenizer.encode(self.REVIEW)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = self.model.generate(input_ids, 
                        max_length=self.max_length_token, 
                        repetition_penalty=1.2, 
                        do_sample=True)
        output = self.tokenizer.decode(output[0], skip_special_tokens=False)
        processed_output = postprocess(output, store_name=store_name, customer_name=customer_name)
        return processed_output

