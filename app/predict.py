# import streamlit as st
from typing import List, Tuple
import torch
import torch.nn as nn
from app.model import KoBART, KoGPT2
import streamlit as st


def load_model(model_type: str = 'KoBART', model_path: str = None) -> nn.Module:
    if model_type == 'KoBART':
        model = KoBART(model_path=model_path)
    
    elif model_type == 'KoGPT2':
        model = KoGPT2()
        if model_path:
            model.load_state_dict(torch.load(model_path))
    return model

def predict_from_model(model: nn.Module, input_str, store_name, customer_name) -> str:
    output_string = model.forward(input_str, store_name, customer_name)
    return output_string
