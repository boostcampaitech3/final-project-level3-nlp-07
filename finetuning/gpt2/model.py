import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

class KoGPTConditionalGeneration(nn.Module):
    def __init__(self):
        super(KoGPTConditionalGeneration, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.model.train()

        self.pad_token_id = 0
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                       bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                       pad_token='<pad>', mask_token='<mask>') 
    
        self.neg = 0

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,):
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        return output
        

    # def training_step(self, batch, batch_idx):
    #     token_ids = batch['input']
    #     mask = batch['mask']
    #     label = batch['labels']

    #     out = self(token_ids)        
    #     mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
    #     mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
    #     loss = self.loss_function(mask_out.transpose(2, 1), label)
    #     loss_avg = loss.sum() / mask.sum()
    #     self.log('train_loss', loss_avg)
    #     return loss_avg

    # def validation_step(self, batch, batch_idx):
    #     token_ids = batch['input']
    #     mask = batch['mask']
    #     label = batch['labels']
        
    #     out = self(token_ids)
    #     mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
    #     mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
    #     loss = self.loss_function(mask_out.transpose(2, 1), label)
    #     loss_avg = loss.sum() / mask.sum()
    #     return (loss_avg)

    # def validation_epoch_end(self, outputs):
    #     losses = []
    #     for loss in outputs:
    #         losses.append(loss)
    #     self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)