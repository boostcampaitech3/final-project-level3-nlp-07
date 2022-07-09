# uvicorn predict:app --reload
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from utils import generate_next_token
from fastapi import FastAPI
import streamlit as st
from transformers import BartForConditionalGeneration
from train import seed_everything, KoGPTConditionalGeneration
import yaml

app = FastAPI()

# 모델 load
seed_everything(42)

@st.cache(allow_output_mutation=True)
def load_kobart():
    return BartForConditionalGeneration.from_pretrained('./models/kobart')
@st.cache(allow_output_mutation=True)
def load_kogpt2():
    with open('./models/kogpt2/hparams.yaml') as f:
        hparams = yaml.load(f)
    return KoGPTConditionalGeneration.load_from_checkpoint('./models/kogpt2/epoch=04.ckpt', hparams=hparams)

model_kobart = load_kobart()
model_kogpt2 = load_kogpt2()

print('koBART model is loaded! {}'.format(model_kobart is not None))
print('koGPT2 model is loaded! {}'.format(model_kogpt2 is not None))


@app.get('/')
async def hello_world():
    return {"hello": "world"}

@app.post("/generate_from_kobart", description="koBART 모델로 부터 사장답글 생성합니다")
async def inference_from_kobart(input_str, store_name, customer_name):
    model = model_kobart
    input_str = input_str.replace('\n', '')
    special_tokens = ['#@상호명#', '#@고객이름#', '#@위치#', '#@기관#', '#@전화번호#']

    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2", special_tokens=special_tokens)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    input_ids = tokenizer.encode(input_str, return_tensors='pt')
    output = model.generate(input_ids, 
                            eos_token_id=1, 
                            max_length=128, 
                            num_beams=5, 
                            no_repeat_ngram_size=2, 
                            early_stopping=True)

    output = tokenizer.decode(output[0], skip_special_tokens=False)
    processed_output = postprocess(output, store_name=store_name, customer_name=customer_name)
    return processed_output

@app.post("/generate_from_kogpt2", description="koGPT2 모델로 부터 사장답글 생성합니다")
async def inference_from_kogpt2(input_str, store_name, customer_name):
    model = model_kogpt2
    SUMMARY = '<unused1>'
    BOS = '</s>'
    EOS = '</s>'
    PTUNING = '<unused2>'

    input_str = input_str.replace('\n', '')
    tokenizer = model.tokenizer
    input_tokens = tokenizer.encode(PTUNING)* 10 + tokenizer.encode(input_str) + tokenizer.encode(SUMMARY)
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)
    eos_id = tokenizer.encode(EOS)[0]

    limit = 0
    while True:
        if limit > 200:
            break
        limit += 1
        pred = model.model(input_tensor)
        next_token = generate_next_token(pred.logits, temperature=1.0, top_p=0.8)

        if next_token.item() == eos_id:
            break
        else:
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)],1)
    predict = tokenizer.decode(input_tensor[0]).split('<unused1>')[-1].strip()
    processed_output = postprocess(predict, store_name=store_name, customer_name=customer_name)
    return processed_output

def postprocess(x, store_name, customer_name):
    output_x = x.replace("[a-zA-Z]+", "")
    output_x = output_x.replace("#@상호명#", store_name)
    output_x = output_x.replace("#@고객이름#", customer_name)
    output_x = output_x.replace("#@위치#", "")
    output_x = output_x.replace("<unk>", "")
    output_x = output_x.replace("</s>", "")
    output_x = output_x.replace("\n", " ")
    return output_x
