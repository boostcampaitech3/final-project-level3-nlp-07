import torch.nn as nn
from app.model import KoBART, KoGPT2
from app.train import seed_everything


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
