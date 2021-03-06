# streamlit run main.py --server.port 30001 --server.fileWatcherType none
import streamlit as st
import requests

st.set_page_config(page_title="카페 사장답글 생성 서비스", page_icon="☕️", layout="centered")

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

    
    st.markdown("## 사장님 답글 제안 (koBART)")
    with st.spinner("Generating..."):
        params = {
            "input_str": input_str_kobart,
            "store_name": store_name,
            "customer_name": customer_name
        }
        text_kobart = requests.post("http://localhost:8000/generate_from_kobart", params=params)
        st.write(text_kobart.text)

    st.markdown("## 사장님 답글 제안 (koGPT2)")
    with st.spinner("Generating..."):
        params = {
            "input_str": input_str_kogpt2,
            "store_name": store_name,
            "customer_name": customer_name
        }
        text_kogpt2 = requests.post("http://localhost:8000/generate_from_kogpt2", params=params)
        st.write(text_kogpt2.text)


st.markdown("Developed by 참께 자라기 For Naver Boost Camp NLP 07 Final Project", unsafe_allow_html=True) 