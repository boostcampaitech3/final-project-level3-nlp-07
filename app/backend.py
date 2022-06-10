from fastapi import FastAPI
from typing import List
from app.predict import load_model, predict_from_model

app = FastAPI()

history = []


@app.get("/")
def hello_world():
    return {"hello": "world"}


@app.get("/history", description="생성한 문장들 기록을 가져옵니다")
async def get_history() -> List[str]:
    return history


@app.post("/generate_from_kobart", description="입력에 대한 사장답글 생성합니다")
async def get_generated_string(input_str, store_name, customer_name):

    model_kobart = load_model(model_type='KoBART', model_path='./finetuning/kobart/best_model/')
    generated_string = predict_from_model(model=model_kobart, 
                        input_str=input_str, 
                        store_name=store_name,
                        customer_name=customer_name)

    history.append(generated_string)
    return generated_string


@app.post("/generate_from_kogpt2", description="입력에 대한 사장답글 생성합니다")
async def get_generated_string(input_str, store_name, customer_name):

    model_kogpt2 = load_model(model_type='KoGPT2', 
                                model_path='./finetuning/kogpt2/epoch=04.ckpt', 
                                hparams_file='./finetuning/kogpt2/hparams.yaml')
    generated_string = predict_from_model(model=model_kogpt2, 
                        input_str=input_str, 
                        store_name=store_name,
                        customer_name=customer_name)

    history.append(generated_string)
    return generated_string
