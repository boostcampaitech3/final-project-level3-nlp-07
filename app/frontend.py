import requests
import streamlit as st

st.set_page_config(page_title="카페 사장답글 생성 서비스", page_icon="☕️", layout="centered")

st.title("☕️ 배달 앱 리뷰 답글 생성봇")

st.markdown("안녕하세요! 저는 카페 사장님들의 빛과 소금이 될 배달 앱 리뷰 답글 생성봇 입니다!", unsafe_allow_html=True)

with st.form(key="customer_form", clear_on_submit=True):
    store_name = st.text_input("점포명", placeholder="점포명을 입력해주세요.")
    customer_name = st.text_input("손님 이름", placeholder="손님 이름을 입력해주세요.")

    star_cols = st.columns(3)
    taste_star = star_cols[0].radio("맛 별점", ("1", "2", "3", "4", "5", "별점X"))
    quantity_star = star_cols[1].radio("양 별점", ("1", "2", "3", "4", "5", "별점X"))
    delivery_star =  star_cols[2].radio("배달 별점", ("1", "2", "3", "4", "5", "별점X"))
    customer_review = st.text_area("손님 리뷰", placeholder="손님 리뷰를 입력해주세요.", key="text")

    submit = st.form_submit_button(label="사장 답글 생성")


if customer_review:
    punct = ('!', '?', '.')
    input_str = f'맛:{taste_star} 양:{quantity_star} 배달:{delivery_star}'

    params_dict = {
        "input_str": input_str, 
        "store_name": store_name,
        "customer_name": customer_name
    }
    
    with st.spinner("Generating..."):
        st.header("사장님 답글")

        response_kobart = requests.post("http://localhost:8001/generate_from_kobart", params=params_dict)
        st.write(response_kobart.text)

        response_kogpt2 = requests.post("http://localhost:8001/generate_from_kogpt2", params=params_dict)
        st.write(response_kogpt2.text)

