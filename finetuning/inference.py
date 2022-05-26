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
from load_data import *
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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)




def inference(model, tokenized_sent, device, tokenizer):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    
    
    rouge = Rouge(metrics=['rouge-n','rouge-l','rouge-w'])

    rouge_l = [0,0,0]   #각각 f,p,r
    rouge_w = [0,0,0]   #각각 f,p,r


    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_label = []
    
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=data['input_ids'].to(device), max_length=512
                )

        # print(outputs.shape)
        # print(data['labels'])
        
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(data['labels'], skip_special_tokens=True)

        output_pred.extend(output_str)
        output_label.extend(label_str)

        break
    

    #tqdm()은 단지 코드 수행 과정을 가시적으로 보기 위함이라 신경쓰지 않으셔도 됩니다.
    for i in tqdm(range(len(output_pred)),desc="(평가중...)",ascii=True):
        sumR = output_pred[i]
        labelR = output_label[i]
        
        result = rouge.get_scores(sumR,labelR)

        rouge_l[0] += float(result['rouge-l']['f'])
        rouge_l[1] += float(result['rouge-l']['p'])
        rouge_l[2] += float(result['rouge-l']['r'])
        rouge_w[0] += float(result['rouge-w']['f'])
        rouge_w[1] += float(result['rouge-w']['p'])
        rouge_w[2] += float(result['rouge-w']['r'])

        for j in range(len(rouge_l)):
            rouge_l[j] /= (i+1)
            rouge_w[j] /= (i+1)

    return rouge_l, rouge_w


def main(args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    seed_everything(42)
    # load model and tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2', return_special_tokens_mask = True)

    # load dataset
    dataset = pd.read_csv("total_data.csv", encoding='utf-8')

    train_dataset, dev_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # make dataset for pytorch.
    RE_train_dataset = CustomTestDataset(train_dataset, tokenizer, max_len=256)
    RE_dev_dataset = CustomTestDataset(dev_dataset, tokenizer, max_len=256)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    ## load my model
    model = BartForConditionalGeneration.from_pretrained('./results/checkpoint-500')
    model.resize_token_embeddings(len(tokenizer))
    model.parameters
    model.to(device)

    ## predict answer
    rouge_l, rouge_w = inference(model, RE_dev_dataset, device, tokenizer) # model에서 class 추론
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
  