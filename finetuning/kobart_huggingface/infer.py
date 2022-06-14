import torch
import streamlit as st
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

@st.cache
def load_model():
    model = BartForConditionalGeneration.from_pretrained('./best_model')
    # model = BartForConditionalGeneration.from_pretrained('./test_epoch3')
    return model

model = load_model()
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
special_tokens_dict = {'additional_special_tokens': ['#@상호명#', '#@위치#', '#@기관#', '#@고객이름#', '#@전화번호#']}
tokenizer.add_special_tokens(special_tokens_dict)

st.title("KoBART 사장님 답글 생성 Test")
text = st.text_area("손님 리뷰 입력:")

st.markdown("## 손님 리뷰 원문")
st.write(text)

if text:
    text = text.replace('\n', '')
    st.markdown("## KoBART 사장님 답글 생성 결과")
    with st.spinner('processing..'):
        input_ids = tokenizer.encode('맛:5 양:5 배달:5 리뷰:' + text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids, eos_token_id=1, max_length=128, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        # output = model.generate(input_ids, eos_token_id=1, max_length=64, repetition_penalty=1.2, do_sample=True, early_stopping=True)
        output = tokenizer.decode(output[0], skip_special_tokens=False)
    st.write(output)