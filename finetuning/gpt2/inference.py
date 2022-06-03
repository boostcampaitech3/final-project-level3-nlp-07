import pickle as pickle
import torch
import random
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer, Trainer, TrainingArguments
import wandb
import argparse
from importlib import import_module
from load_dataset import *
from datasets import load_metric
#encoding=utf-8
from transformers import (    
    BartForConditionalGeneration,PreTrainedTokenizerFast,

  )
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
import torch
from torch.utils.data import random_split
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset, load_metric
from tqdm import tqdm
from torch.utils.data import DataLoader
from rouge import *
from model import KoGPTConditionalGeneration
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from load_dataset import *
import torch.nn.functional as F

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def generate_next_token(logits, temperature=1.0, top_k=0, top_p=0.9):
    logits = logits[0, -1, :] / temperature
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probabilities = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probabilities, 1)
    return next_token

def inference(model, customer, store, dataset, device, tokenizer):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    REVIEW = '<unused1>' # review의 시작 부근
    PTUNING = '<unused2>'
    BOS = '<bos>'
    EOS = '<eos>'
    
    # rouge = Rouge(metrics=['rouge-n','rouge-l','rouge-w'])

    # rouge_l = [0,0,0]   #각각 f,p,r
    # rouge_w = [0,0,0]   #각각 f,p,r


    #cdataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_label = []

    # beam search https://huggingface.co/blog/how-to-generate
    # for i, data in enumerate(tqdm(customer)):
    #     text = data.replace('\n', '')
    #     input_tokens = tokenizer.encode(PTUNING)* 10 + tokenizer.encode(text) + tokenizer.encode(REVIEW)
    #     input_tensor = torch.tensor(input_tokens).unsqueeze(0).to('cuda')

    #     eos_id = tokenizer.encode(EOS)[0]

    #     label = store[i].replace('\n', '')
        
    #     while True:
    #         pred = model(input_tensor)
    #         next_token = generate_next_token(pred.logits, temperature=1.0, top_p=0.8).to('cuda')
    #         if (next_token.item() == eos_id) or (input_tensor.shape[1] > 100):
    #             break
    #         else:
    #             input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)],1).to('cuda')
        
    #     output = tokenizer.decode(input_tensor[0], skip_special_tokens=True)
    #     print("output:", output)
    #     print("label: ", label)
        
    #     break
        
    for i, data in enumerate(tqdm(customer)):
        print('text:', data)
        input_tokens = tokenizer.encode(BOS) + tokenizer.encode(data) + tokenizer.encode(REVIEW)
        input_tensor = torch.tensor(input_tokens).unsqueeze(0).to('cuda')

        outputs = model.generate(
                input_ids=input_tensor,
                max_length=50, repetition_penalty=1.2, do_sample=True, num_return_sequences=3)

        output = tokenizer.decode(outputs[0])#, skip_special_tokens=True)
        ret = re.sub(r'(<s>|</s>)', '' , ''.join(output).replace('▁', ' ').strip())
        print('Generated {}: {}'.format(i, ret))

        label = store[i].replace('\n', '')
        print('label {}: {}'.format(i, label))
        print('='*30)
    
    for i in tqdm(range(len(output_pred)),desc="(평가중...)",ascii=True):
        
        result = rouge.get_scores(output_pred[i], output_label[i])

        rouge_l[0] += float(result['rouge-l']['f'])
        rouge_l[1] += float(result['rouge-l']['p'])
        rouge_l[2] += float(result['rouge-l']['r'])
        rouge_w[0] += float(result['rouge-w']['f'])
        rouge_w[1] += float(result['rouge-w']['p'])
        rouge_w[2] += float(result['rouge-w']['r'])

        for j in range(len(rouge_l)):
            rouge_l[j] /= (i+1)
            rouge_w[j] /= (i+1)
        
        print(rouge_l)    
        break

    return rouge_l, rouge_w


def main(args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    seed_everything(42)
    # load model and tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                       bos_token='<bos>', eos_token='<eos>', unk_token='<unk>',
                       pad_token='<pad>', mask_token='<mask>', add_special_tokens=['#@상호명#', '#@위치#', '#@기관#']) 

    # load dataset
    test_dataset = pd.read_csv("test2.csv", encoding='utf-8')
    customer = test_dataset['고객리뷰'].tolist()
    store = test_dataset['사장답글'].tolist()
    
    
    # make dataset for pytorch.
    RE_test_dataset = KoGPTSummaryTestDataset(test_dataset, tokenizer, max_len=300)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## load my model    
    model = GPT2LMHeadModel.from_pretrained('./results/checkpoint-9500')
    #model.load_state_dict(torch.load('/opt/ml/final-project-level3-nlp-07/finetuning/gpt2/full_reviews/checkpoint-16000/pytorch_model.bin'))
    model.to(device)

    ## predict answer
    rouge_l, rouge_w = inference(model, customer, store, RE_test_dataset, device, tokenizer) # model에서 class 추론
    print(rouge_l, rouge_w)


    print('---- Finish! ----')
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--test_dataset', type=str, default="../dataset/test/test_data.csv")
  parser.add_argument('--model_dir', type=str, default="./best_model")

  # load_data module
  parser.add_argument('--load_data_filename', type=str, default="load_data")
  parser.add_argument('--load_data_func_load', type=str, default="load_data")
  parser.add_argument('--load_data_func_tokenized', type=str, default="tokenized_dataset")
  parser.add_argument('--load_data_class', type=str, default="RE_Dataset")

  args = parser.parse_args()
  print(args)
  main(args)
  