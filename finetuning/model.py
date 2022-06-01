import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedTokenizerFast, BartForConditionalGeneration
from torch.cuda.amp import autocast
from typing import List, Optional, Tuple, Union

class KoBARTConditionalGeneration(nn.Module):
    def __init__(self):
        super(KoBARTConditionalGeneration, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,):

        attention_mask = input_ids.ne(self.pad_token_id).float()
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).float()
        
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          decoder_attention_mask=decoder_attention_mask,
                          labels=labels, return_dict=True)



    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     outs = self(batch)
    #     loss = outs['loss']
    #     return (loss)

    # def validation_epoch_end(self, outputs):
    #     losses = []
    #     for loss in outputs:
    #         losses.append(loss)
    #     self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)
