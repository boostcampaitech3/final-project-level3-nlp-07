# streamlit run app/main.py --server.port 30001 --server.fileWatcherType none
import streamlit as st
from transformers import BartForConditionalGeneration
from train import KoGPTConditionalGeneration
from train import seed_everything
from predict import inference_from_kobart, inference_from_kogpt2
import yaml


st.set_page_config(page_title="카페 사장답글 생성 서비스", page_icon="☕️", layout="centered")

# 모델 load
seed_everything(42)

@st.cache(allow_output_mutation=True)
def load_kobart():
    model_kobart = BartForConditionalGeneration.from_pretrained('./finetuning/kobart/best_model')
    return model_kobart

@st.cache(allow_output_mutation=True)
def load_kogpt2():
    with open('./finetuning/kogpt2/hparams.yaml') as f:
        hparams = yaml.load(f)
    model_kogpt2 =  KoGPTConditionalGeneration.load_from_checkpoint('./finetuning/kogpt2/epoch=04.ckpt', hparams=hparams)
    return model_kogpt2

model_kobart = load_kobart()
model_kogpt2 = load_kogpt2()

st.title("☕️ 배달 앱 리뷰 답글 생성봇")

st.markdown("안녕하세요! 저는 카페 사장님들의 빛과 소금이 될 배달 앱 리뷰 답글 생성봇 입니다!", unsafe_allow_html=True)


with st.form(key="customer_form", clear_on_submit=False):
    store_name = st.text_input("점포명", placeholder="점포명을 입력해주세요.")
    customer_name = st.text_input("손님 이름", placeholder="손님 이름을 입력해주세요.")

    star_cols = st.columns(3)
    taste_star = star_cols[0].radio("맛 별점", ("5", "4", "3", "2", "1", "X"))
    quantity_star = star_cols[1].radio("양 별점", ("5", "4", "3", "2", "1", "X"))
    delivery_star =  star_cols[2].radio("배달 별점", ("5", "4", "3", "2", "1", "X"))
    customer_review = st.text_area("손님 리뷰", placeholder="손님 리뷰를 입력해주세요.", key="text")

    submit = st.form_submit_button(label="사장 답글 생성")


if submit:
    input_str_kobart = f'맛:{taste_star} 양:{quantity_star} 배달:{delivery_star} 리뷰:{customer_review}'
    input_str_kogpt2 = f'리뷰: {customer_review}'

    # string 만 넘겨줄 수 있음
    
    st.markdown("## 사장님 답글 제안 (koBART)")
    with st.spinner("Generating..."):
        text_kobart = inference_from_kobart(model_kobart, input_str_kobart, store_name, customer_name)
        st.write(text_kobart)

    st.markdown("## 사장님 답글 제안 (koGPT2)")
    with st.spinner("Generating..."):
        text_kogpt2 = inference_from_kogpt2(model_kogpt2, input_str_kogpt2, store_name, customer_name)
        st.write(text_kogpt2)


st.markdown("Developed by 참께 자라기 For Naver Boost Camp NLP 07 Final Project", unsafe_allow_html=True) 