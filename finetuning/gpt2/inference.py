import pickle as pickle
import torch
import random
import sklearn
import numpy as np
import pandas as pd
import wandb
import argparse
from load_dataset import *
#encoding=utf-8
from transformers import PreTrainedTokenizerFast
import torch
import pandas as pd
from datasets import load_metric
from tqdm import tqdm
from rouge import *
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

def inference(model, dataset, device, tokenizer, type):
    REVIEW = '<unused1>' # review의 시작 부근
    BOS = '<bos>'
    EOS = '<eos>'

    model.eval()
    input = []
    output_pred = []
    output_label = []
    
    for i, _ in enumerate(tqdm(range(5))):#len(dataset)))):
        input.append(dataset['고객리뷰'].iloc[i])
        input_tokens = tokenizer.encode(BOS) + tokenizer.encode("맛: ") + tokenizer.encode(dataset['맛'].iloc[i]) + \
                        tokenizer.encode("양: ") + tokenizer.encode(dataset['양'].iloc[i]) +  tokenizer.encode("배달: ") + tokenizer.encode(dataset['배달'].iloc[i]) + \
                        tokenizer.encode("리뷰: ") + tokenizer.encode(dataset['고객리뷰'].iloc[i]) + tokenizer.encode(REVIEW)
                    
                    
        input_tensor = torch.tensor(input_tokens).unsqueeze(0).to('cuda')

        if type == "sampling":
            outputs = model.generate(
                    input_ids=input_tensor,
                    max_length=128, repetition_penalty=1.2, do_sample=True)
        elif type == "greedy":
            outputs = model.generate(input_ids=input_tensor, max_length=128)
        elif type == "beam":
            outputs = model.generate(input_ids=input_tensor, max_length=128, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

        output = tokenizer.decode(outputs[0]).split('<unused1>')[-1].strip().split('<eos>')[0].strip()
        print(output)
        output_pred.append(output)

        label = dataset['사장답글'].iloc[i].replace('\n', '')
        output_label.append(label)


    df = pd.DataFrame({'고객리뷰': input, '예측답글': output_pred, '사장답글': output_label})
    df.to_csv("emotion_menu_result.csv", encoding='utf-8')



def main(args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    seed_everything(42)
    # load model and tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                        bos_token='<bos>', eos_token='<eos>', unk_token='<unk>',
                        pad_token='<pad>', mask_token='<mask>') 
    
    special_tokens_dict = {'additional_special_tokens': ['#@상호명#', '#@위치#', '#@기관#', '#@고객이름#', '#@전화번호#']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # load dataset
    test_dataset = pd.read_csv("test.csv", encoding='utf-8')


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## load my model    
    model = GPT2LMHeadModel.from_pretrained('./with_menu/checkpoint-15000')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    ## predict answer
    inference(model, test_dataset, device, tokenizer, args.decode) # model에서 class 추론
    


    print('---- Finish! ----')
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--test_dataset', type=str, default="../dataset/test/test_data.csv")
  parser.add_argument('--model_dir', type=str, default="./best_model")

  # load_data module
  parser.add_argument('--decode', type=str, default="greedy")

  args = parser.parse_args()
  print(args)
  main(args)
  