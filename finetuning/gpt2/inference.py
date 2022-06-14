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

def inference(model, customer, store, menu, emotion, device, tokenizer):
    REVIEW = '<unused1>' # review의 시작 부근
    BOS = '<bos>'
    EOS = '<eos>'
    
    rouge = Rouge(metrics=['rouge-n','rouge-l','rouge-w'])

    rouge_l = [0,0,0]   #각각 f,p,r
    rouge_w = [0,0,0]   #각각 f,p,r

    model.eval()
    input = []
    output_pred = []
    output_label = []
    
    for i, data in enumerate(tqdm(customer)):
        input.append(data)
        input_tokens = tokenizer.encode(BOS) + tokenizer.encode(menu[i]) + tokenizer.encode(data) + tokenizer.encode(REVIEW)
        input_tensor = torch.tensor(input_tokens).unsqueeze(0).to('cuda')

        outputs = model.generate(
                input_ids=input_tensor,
                max_length=256, repetition_penalty=1.2, do_sample=True)

        output = tokenizer.decode(outputs[0])#, skip_special_tokens=True)
        print(output)
        #ret = re.sub(r'(<s>|</s>)', '' , ''.join(output).replace('▁', ' ').strip())
        output_pred.append(output)
        # print('Generated {}: {}'.format(i, ret))

        label = store[i].replace('\n', '')
        output_label.append(label)
        # print('label {}: {}'.format(i, label))
        # print('='*30)
        
    
    # for i in tqdm(range(len(output_pred)),desc="(평가중...)",ascii=True):
        
    #     result = rouge.get_scores(output_pred[i], output_label[i])

    #     rouge_l[0] += float(result['rouge-l']['f'])
    #     rouge_l[1] += float(result['rouge-l']['p'])
    #     rouge_l[2] += float(result['rouge-l']['r'])
    #     rouge_w[0] += float(result['rouge-w']['f'])
    #     rouge_w[1] += float(result['rouge-w']['p'])
    #     rouge_w[2] += float(result['rouge-w']['r'])

    #     for j in range(len(rouge_l)):
    #         rouge_l[j] /= (i+1)
    #         rouge_w[j] /= (i+1)
        

    df = pd.DataFrame({'input': input, 'predicted': output_pred, 'label': output_label})
    df.to_csv("emotion_menu_result.csv", encoding='utf-8')
    return rouge_l, rouge_w


def main(args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    seed_everything(42)
    # load model and tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                        bos_token='<bos>', eos_token='<eos>', unk_token='<unk>',
                        pad_token='<pad>', mask_token='<mask>') 
    
    special_tokens_dict = {'additional_special_tokens': ['#@상호명#', '#@위치#', '#@기관#']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # load dataset
    test_dataset = pd.read_csv("test.csv", encoding='utf-8')
    customer = test_dataset['고객리뷰'].tolist()
    store = test_dataset['사장답글'].tolist()
    menu = test_dataset['주문메뉴'].tolist()
    emotion = test_dataset['공통감정'].tolist()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## load my model    
    model = GPT2LMHeadModel.from_pretrained('./with_menu/checkpoint-15000')
    model.resize_token_embeddings(len(tokenizer))
    #model.load_state_dict(torch.load('/opt/ml/final-project-level3-nlp-07/finetuning/gpt2/full_reviews/checkpoint-16000/pytorch_model.bin'))
    model.to(device)

    ## predict answer
    rouge_l, rouge_w = inference(model, customer, store, menu, emotion, device, tokenizer) # model에서 class 추론
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
  