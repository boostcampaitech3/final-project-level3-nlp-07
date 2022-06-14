import torch
import random
import numpy as np
import argparse
from transformers import (    
    BartForConditionalGeneration,PreTrainedTokenizerFast,

  )
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    seed_everything(42)
    # load model and tokenizer

    model_bart = BartForConditionalGeneration.from_pretrained('./best_model')

    special_tokens_dict = {'additional_special_tokens': ['#@상호명#', '#@위치#', '#@기관#', '#@고객이름#', '#@전화번호#']}
    tokenizer_bart = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    tokenizer_bart.add_special_tokens(special_tokens_dict)

    model_bart.resize_token_embeddings(len(tokenizer_bart))

    test_dataset = load_from_disk('./datasets/test_dataset')

    # 전처리 함수
    def preprocess_function(examples):
        inputs = ["맛:" + i + " 양:" + j + " 배달:" + k + " 리뷰:" + m for i, j, k, m in zip(
                        examples["맛"],examples["양"],examples["배달"],
                        examples["고객리뷰"])]
        model_inputs = tokenizer_bart(inputs, max_length=256, truncation=True)

        with tokenizer_bart.as_target_tokenizer():
            labels = tokenizer_bart(examples["사장답글"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    test_dataset = test_dataset.map(preprocess_function, batched=True)

    outputs = []

    for input_text in tqdm(test_dataset['input_ids']):
        if args.command == 'greedy':
            output_bart = model_bart.generate(torch.tensor(input_text).unsqueeze(0), max_length=128)
        elif args.command == 'sampling':
            output_bart = model_bart.generate(torch.tensor(input_text).unsqueeze(0), max_length=128, repetition_penalty=1.2, do_sample=True)
        else:
            output_bart = model_bart.generate(torch.tensor(input_text).unsqueeze(0), eos_token_id=1, max_length=128, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        
        
        temp = tokenizer_bart.decode(output_bart[0], skip_special_tokens=False)

        temp = temp.replace('</s>', '')
        temp = temp.replace('<unk>', '')
        temp = temp.strip()

        outputs.append(temp)

    temp_df = pd.DataFrame()

    temp_df['고객리뷰'] = test_dataset['고객리뷰']
    temp_df['사장답글'] = test_dataset['사장답글']
    temp_df['예측답글'] = outputs

    temp_df.to_csv(args.command+'.csv', encoding='utf-8', index=False)


    print('---- Finish! ----')
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--command', type=str, default="greedy")

  args = parser.parse_args()
  print(args)
  main(args)
  