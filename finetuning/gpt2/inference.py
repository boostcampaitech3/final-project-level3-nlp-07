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

def inference(model, dataset, device, tokenizer, type):
    REVIEW = '<unused1>' # review의 시작 부근
    BOS = '<bos>'
    EOS = '<eos>'

    model.eval()
    input = []
    output_pred = []
    output_label = []
    
    for i, _ in enumerate(tqdm(range(len(dataset)))):
        input.append(dataset['고객리뷰'].iloc[i])
        input_tokens = tokenizer.encode(BOS) + tokenizer.encode("맛: ") + tokenizer.encode(dataset['맛'].iloc[i]) + \
                        tokenizer.encode("양: ") + tokenizer.encode(dataset['양'].iloc[i]) +  tokenizer.encode("배달: ") + tokenizer.encode(dataset['배달'].iloc[i]) + \
                        tokenizer.encode("리뷰: ") + tokenizer.encode(dataset['고객리뷰'].iloc[i]) + tokenizer.encode(REVIEW)


        input_tensor = torch.tensor(input_tokens).unsqueeze(0).to('cuda')
        eos_id = tokenizer.encode(EOS)[0]

        if type == "sampling":
            outputs = model.generate(
                    input_ids=input_tensor,
                    max_length=256, repetition_penalty=1.2, do_sample=True)
        elif type == "greedy":
            outputs = model.generate(input_ids=input_tensor, max_length=128)
        elif type == "beam":
            outputs = model.generate(input_ids=input_tensor, max_length=128, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        elif type == "custom":
            while True:
                pred = model(input_tensor)
                next_token = generate_next_token(pred.logits, temperature=1.0, top_p=0.8)

                if next_token.item() == eos_id:
                        break
                else:
                    input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)],1)
            outputs = input_tensor

        # output = tokenizer.decode(outputs[0])#.split('<unused1>')[-1].strip().split('<eos>')[0].strip()
        # print(output)
        output = tokenizer.decode(outputs[0]).split('<unused1>')[-1].strip().split('<eos>')[0].strip()
        # print(output)
        output_pred.append(output)

        label = dataset['사장답글'].iloc[i].replace('\n', '')
        output_label.append(label)


    df = pd.DataFrame({'고객리뷰': input, '예측답글': output_pred, '사장답글': output_label})
    df.to_csv("result_%s.csv" % str(type), encoding='utf-8')



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
    model = GPT2LMHeadModel.from_pretrained('./results/checkpoint-4500')
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
  parser.add_argument('--decode', type=str, default="custom")

  args = parser.parse_args()
  print(args)
  main(args)
  