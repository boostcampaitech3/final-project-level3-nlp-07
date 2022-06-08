import os
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader

MASK = '<mask>'
STORE_REVIEW = '<unused1>'
BOS = '<bos>' #0
EOS = '<eos>' #1
PAD = '<pad>' #3

class KoGPTSummaryDataset(Dataset):
    def __init__(self, dataset, tok, max_len=300,
                 bos_token=BOS, eos_token=EOS,
                 pad_token=PAD, mask_token=MASK,
                 store_review_token = STORE_REVIEW,
                 ignore_index = -100,
                 prompt_length = 0
                ):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        self.docs = dataset
        self.len = self.docs.shape[0]
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.store_review_token = store_review_token

        self.ignore_index = ignore_index
        self.prompt_length = prompt_length

    def add_padding_data(self, inputs, pad_index):
        if len(inputs) < self.max_len:
            pad = [pad_index] *(self.max_len - len(inputs))
            inputs = inputs + pad
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        
        article = self.tok.encode(self.bos_token) + self.tok.encode(instance['주문메뉴']) + self.tok.encode(instance['고객리뷰'])
        len_article = len(article)

        summary = self.tok.encode(self.store_review_token) + self.tok.encode(instance['사장답글']) + self.tok.encode(self.eos_token)
        len_summary = len(summary)

        context = article + summary
        
        if len(context) > self.max_len:
            additional_len = len(context) - self.max_len
            article = article[:-additional_len]
            len_article = len(article)
            context = article + summary

        labels = summary
        mask = [0] * len_article + [1] * len_summary + [0] * (self.max_len - len_article - len_summary)


        if len(context) < self.max_len:
            context = self.add_padding_data(context, self.tok.encode('<pad>')[0])

        if len(labels) < self.max_len:
            labels = self.add_padding_data(labels, self.tok.encode('<pad>')[0])

        return {'input_ids': np.array(context, dtype=np.int_),
                'attention_mask': np.array(mask, dtype=np.int_),
                'labels': np.array(labels, dtype=np.int_)}

    def __len__(self):
        return self.len
    
 
