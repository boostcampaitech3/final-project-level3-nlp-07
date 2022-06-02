import requests
import streamlit as st


st.set_page_config(layout="wide")

st.title("사장답글 Gereration Model")
customer_review = "크로플이 맛잇으셨군요."

if customer_review:
    input_dict = {
        "input_string": customer_review
    }
    
    with st.spinner("Generating..."):
        response = requests.post("http://localhost:8001/generate", params=input_dict)
        generated_string = response.text
        
        st.header("사장님 답글")
        st.write(generated_string)

