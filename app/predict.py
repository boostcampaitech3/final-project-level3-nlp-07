# import streamlit as st
import numpy as np
import random
from typing import List, Tuple
import torch
import torch.nn as nn
from app.model import KoBART, KoGPT2
import streamlit as st

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_model(model_type: str = 'KoBART', model_path: str = None, hparams_file: str = None) -> nn.Module:
    seed_everything(42)
    if model_type == 'KoBART':
        model = KoBART(model_path=model_path)
    
    elif model_type == 'KoGPT2':
        model = KoGPT2(hparams_file=hparams_file, model_path=model_path)
    return model

def predict_from_model(model: nn.Module, input_str, store_name, customer_name) -> str:
    output_string = model.forward(input_str, store_name, customer_name)
    return output_string
