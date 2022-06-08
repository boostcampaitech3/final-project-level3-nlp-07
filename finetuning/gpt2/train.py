import torch
import random
import sklearn
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoConfig
import wandb
import argparse
from datasets import load_metric
#encoding=utf-8
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_metric
from model import *
from load_dataset import *


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)




def train(args):
  seed_everything(args.seed)
  # load model and tokenizer
  tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                       bos_token='<bos>', eos_token='<eos>', unk_token='<unk>',
                       pad_token='<pad>', mask_token='<mask>') 
  
  special_tokens_dict = {'additional_special_tokens': ['#@상호명#', '#@위치#', '#@기관#', '#@고객이름#', '#@전화번호#']}
  tokenizer.add_special_tokens(special_tokens_dict)

  # load dataset
  train_dataset = pd.read_csv("train.csv", encoding='utf-8')
  dev_dataset = pd.read_csv("val.csv", encoding='utf-8')
  
  # make dataset for pytorch.
  RE_train_dataset = KoGPTSummaryDataset(train_dataset, tokenizer, max_len=256)
  RE_dev_dataset = KoGPTSummaryDataset(dev_dataset, tokenizer, max_len=256)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  config = AutoConfig.from_pretrained('skt/kogpt2-base-v2', 
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    unk_token_id=tokenizer.unk_token_id,
                                    output_hidden_states=False)
  
  model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', config=config)
  model.resize_token_embeddings(len(tokenizer))

  model.to(device)

  data_collator = DataCollatorForLanguageModeling(
          tokenizer=tokenizer,
          mlm=False
      )

  wandb.init(project=args.project_name, entity=args.entity_name)
  wandb.run.name = args.run_name
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=args.output_dir,                     # output directory
    save_total_limit=args.save_total_limit,         # number of total save model.
    save_steps=args.save_steps,                     # model saving step.
    num_train_epochs=args.num_train_epochs,         # total number of training epochs
    learning_rate=args.learning_rate,               # learning rate
    per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.per_device_eval_batch_size,   # batch size for evaluation
    warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,                # strength of weight decay
    logging_dir=args.logging_dir,                  # directory for storing logs
    logging_steps=args.logging_steps,              # log saving step.
    evaluation_strategy=args.evaluation_strategy,  # evaluation strategy to adopt during training
                                                    # `no`: No evaluation during training.
                                                    # `steps`: Evaluate every `eval_steps`.
                                                    # `epoch`: Evaluate every end of epoch.
    eval_steps = args.eval_steps,                             # evaluation step.
    load_best_model_at_end = args.load_best_model_at_end,     # Whether or not to load the best model found during training at the end of training.
    report_to=args.report_to,                                 # The list of integrations to report the results and logs to.
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
    fp16=True,                # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.     
  )
  
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    data_collator=data_collator,
    #compute_metrics=compute_metrics         # define metrics function
  )


  # train model
  trainer.train()
  wandb.finish()

  trainer.save_model()
  tokenizer.save_pretrained("./tokenizer", legacy_format=False)


def main(args):
  train(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # Data and model checkpoints directories
  parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
  parser.add_argument("--train_data", type=str, default="../dataset/train/train.csv", help="train_data directory (default: ../dataset/train/train.csv)")
  parser.add_argument("--output_dir", type=str, default="./results", help="directory which stores various outputs (default: ./results)")
  parser.add_argument("--save_total_limit", type=int, default=10, help="max number of saved models (default: 5)")
  parser.add_argument("--save_steps", type=int, default=500, help="interval of saving model (default: 500)")
  parser.add_argument("--num_train_epochs", type=int, default=3, help="number of train epochs (default: 20)")
  parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate (default: 5e-5)")
  parser.add_argument("--per_device_train_batch_size", type=int, default=32, help=" (default: 16)")
  parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help=" (default: 16)")
  parser.add_argument("--warmup_steps", type=int, default=500, help=" (default: 500)")
  parser.add_argument("--weight_decay", type=float, default=0.01, help=" (default: 0.01)")
  parser.add_argument("--logging_dir", type=str, default="./logs", help=" (default: ./logs)")
  parser.add_argument("--logging_steps", type=int, default=100, help=" (default: 100)")
  parser.add_argument("--evaluation_strategy", type=str, default="steps", help=" (default: steps)")
  parser.add_argument("--eval_steps", type=int, default=500, help=" (default: 500)")
  parser.add_argument("--load_best_model_at_end", type=bool, default=True, help=" (default: True)")
  parser.add_argument("--save_pretrained", type=str, default="./best_model", help=" (default: ./best_model)")

  # updated
  parser.add_argument('--run_name', type=str, default="final")
  parser.add_argument("--n_splits", type=int, default=1, help=" (default: )")
  parser.add_argument("--test_size", type=float, default=0.1, help=" (default: )")
  parser.add_argument("--project_name", type=str, default="[Final] GPT_lkm", help=" (default: )")
  parser.add_argument("--entity_name", type=str, default="growing_sesame", help=" (default: )")
  parser.add_argument("--report_to", type=str, default="wandb", help=" (default: )")
  parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help=" (default: )")
  


  args = parser.parse_args()
  print(args)

  seed_everything(args.seed)
  
  main(args)