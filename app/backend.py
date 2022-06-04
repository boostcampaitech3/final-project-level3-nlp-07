from fastapi import FastAPI
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from typing import List

import torch.nn as nn
from app.model import get_model, predict_from_model

app = FastAPI()

history = []


@app.get("/")
def hello_world():
    return {"hello": "world"}


@app.get("/history", description="생성한 문장들 기록을 가져옵니다")
async def get_history() -> List[str]:
    return history


@app.post("/generate", description="입력에 대한 사장답글 생성합니다")
async def get_generated_string(input_string: str):
    model = get_model(model_type='KoBART', model_path='./finetuning/kobart_huggingface/best_model/')
    generated_string = predict_from_model(model=model, input_string=input_string)
    
    history.append(generated_string)
    return generated_string

