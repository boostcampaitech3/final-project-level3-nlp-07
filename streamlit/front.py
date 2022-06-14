import torch
import string
import streamlit as st
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, BartForConditionalGeneration
import time

#모델 load할 때 모델 명은 바꿔주셔야 합니다!
@st.cache(allow_output_mutation=True)
def load_koGPT2():
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    return model

@st.cache
def load_koBart():
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
    return model

special_tokens_dict = {'additional_special_tokens': ['#@상호명#', '#@위치#', '#@기관#', '#@고객이름#', '#@전화번호#']}

tokenizer_gpt = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
tokenizer_bart = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')

tokenizer_bart.add_special_tokens(special_tokens_dict)
tokenizer_gpt.add_special_tokens(special_tokens_dict)

model_gpt = load_koGPT2()
model_bart = load_koBart()

REVIEW = '<unused1>'
PTUNING = '<unused2>'
BOS = '<bos>'
EOS = '<eos>'

st.title("배달앱 리뷰 답글 생성 test")

with st.form(key="customer_form", clear_on_submit=True):
    store_name = st.text_input("점포명을 입력해주세요.")

    star_cols = st.columns(3)
    customer_name = st.text_input("고객명을 입력해주세요")
    taste_star = st.radio("맛 별점", ("5", "4", "3", "2", "1", "X"))
    quantity_star = st.radio("양 별점", ("5", "4", "3", "2", "1", "X"))
    delivery_star =  st.radio("배달 별점", ("5", "4", "3", "2", "1", "X"))

    customer_review = st.text_area("손님 리뷰", placeholder="손님 리뷰를 입력해주세요.", key="text")

    submit = st.form_submit_button(label="사장 답글 생성")

review_all = '맛: '+taste_star+' 양: '+quantity_star+' 배달:'+delivery_star+' 리뷰:'+customer_review

if review_all:
    st.markdown("### 사장님 답글 제안 (koGPT2)")
    with st.spinner('제안을 생성하는 중이예요...'):
        input_ids_gpt = tokenizer_gpt.encode(BOS) + tokenizer_gpt.encode(customer_review) + tokenizer_gpt.encode(REVIEW)
        input_ids_gpt = torch.tensor(input_ids_gpt)
        input_ids_gpt = input_ids_gpt.unsqueeze(0)
        output_gpt = model_gpt.generate(input_ids_gpt, max_length=128, repetition_penalty=1.2, do_sample=True)
        output_gpt = tokenizer_gpt.decode(output_gpt[0], skip_special_tokens=False)
        output_gpt.replace('#@상호명#', store_name)
        output_gpt.replace('#@고객이름#', customer_name)
    st.write(output_gpt)

    st.markdown("### 사장님 답글 제안 (koBart)")
    with st.spinner('제안을 생성하는 중이예요...'):
        input_ids_bart = tokenizer_bart.encode(review_all)
        input_ids_bart = torch.tensor(input_ids_bart)
        input_ids_bart = input_ids_bart.unsqueeze(0)
        output_bart = model_bart.generate(input_ids_bart, eos_token_id=1, max_length=128, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        output_bart = tokenizer_bart.decode(output_bart[0], skip_special_tokens=False)
        output_bart.replace('#@상호명#', store_name)
        output_bart.replace('#@고객이름#', customer_name)
    st.write(output_bart)

