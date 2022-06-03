import torch
import streamlit as st
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

@st.cache
def load_model():
    model = GPT2LMHeadModel.from_pretrained('./best_model')
    return model

model = load_model()
tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')
st.title("KoGPT2 사장님 답글 생성 Test")
text = st.text_area("손님 리뷰 입력:")

st.markdown("## 손님 리뷰 원문")
st.write(text)

if text:
    text = text.replace('\n', '')
    st.markdown("## KoGPT2 사장님 답글 생성 결과")
    with st.spinner('processing..'):
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids, eos_token_id=1, max_length=128, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(output)