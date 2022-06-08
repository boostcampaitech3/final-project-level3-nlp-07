import torch
import streamlit as st
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, GPT2LMHeadModel

@st.cache
def load_model_bart():
    model = BartForConditionalGeneration.from_pretrained('./best_model')
    return model

@st.cache
def load_model_gpt():
    model = GPT2LMHeadModel.from_pretrained('./checkpoint-15000')
    return model

REVIEW = '<unused1>'
PTUNING = '<unused2>'
BOS = '<bos>'
EOS = '<eos>'

special_tokens_dict = {'additional_special_tokens': ['#@상호명#', '#@위치#', '#@기관#', '#@고객이름#', '#@전화번호#']}

model_bart = load_model_bart()
tokenizer_bart = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
tokenizer_bart.add_special_tokens(special_tokens_dict)

model_gpt = load_model_gpt()
tokenizer_gpt = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')
tokenizer_gpt.add_special_tokens(special_tokens_dict)

st.title("배달앱 사장님 답글 생성 Test")
text = st.text_area("손님 리뷰 입력:")

st.markdown("## 손님 리뷰 원문")
st.write(text)

if text:
    text = text.replace('\n', '')

    st.markdown("## KoGPT-2 사장님 답글 생성 결과")
    with st.spinner('processing..'):
        input_ids_gpt = tokenizer_gpt.encode(BOS) + tokenizer_gpt.encode(text) + tokenizer_gpt.encode(REVIEW)
        input_ids_gpt = torch.tensor(input_ids_gpt)
        input_ids_gpt = input_ids_gpt.unsqueeze(0)
        output_gpt = model_gpt.generate(input_ids_gpt, max_length=128, repetition_penalty=1.2, do_sample=True)
        output_gpt = tokenizer_gpt.decode(output_gpt[0], skip_special_tokens=False)
    st.write(output_gpt)

    st.markdown("## KoBART 사장님 답글 생성 결과")
    with st.spinner('processing..'):
        input_ids_bart = tokenizer_bart.encode('맛:5 양:5 배달:5 리뷰:' + text)
        input_ids_bart = torch.tensor(input_ids_bart)
        input_ids_bart = input_ids_bart.unsqueeze(0)
        output_bart = model_bart.generate(input_ids_bart, eos_token_id=1, max_length=128, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        output_bart = tokenizer_bart.decode(output_bart[0], skip_special_tokens=False)
    st.write(output_bart)
