import torch
import random
import numpy as np
import wandb
import argparse
from datasets import load_from_disk
from transformers import (    
    Trainer,
    TrainingArguments,
    BartForConditionalGeneration,
    BartModel,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
  )
import torch
import pandas as pd

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

  # load dataset
  train_dataset = load_from_disk('./datasets/train_dataset')
  val_dataset = load_from_disk('./datasets/val_dataset')

  # load tokenizer and model for collator
  tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
  model_coll = BartModel.from_pretrained('gogamza/kobart-base-v2')

  # 전처리 함수
  def preprocess_function(examples):
    prefix = ""
    inputs = [prefix + doc for doc in examples["고객리뷰"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["사장답글"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
  
  # 전처리 적용
  train_dataset_tokenized = train_dataset.map(preprocess_function, batched=True)
  val_dataset_tokenized = val_dataset.map(preprocess_function, batched=True)

  # collator 설정
  data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_coll)

  # load model
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
  model.to(device)


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
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
    fp16=True,                # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.     
  )
  
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset_tokenized,         # training dataset
    eval_dataset=val_dataset_tokenized,             # evaluation dataset
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics         # define metrics function
  )


  # train model
  trainer.train()
  wandb.finish()

  model.save_pretrained(args.save_pretrained)

def main(args):
  train(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # Data and model checkpoints directories
  parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
  parser.add_argument("--model", type=str, default="gogamza/kobart-base-v2", help="model to train (default: klue/bert-base)")
  parser.add_argument("--train_data", type=str, default="train.csv", help="train_data directory (default: ../dataset/train/train.csv)")
  parser.add_argument("--output_dir", type=str, default="./results", help="directory which stores various outputs (default: ./results)")
  parser.add_argument("--save_total_limit", type=int, default=5, help="max number of saved models (default: 5)")
  parser.add_argument("--save_steps", type=int, default=500, help="interval of saving model (default: 500)")
  parser.add_argument("--num_train_epochs", type=int, default=1, help="number of train epochs (default: 20)")
  parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate (default: 5e-5)")
  parser.add_argument("--per_device_train_batch_size", type=int, default=32, help=" (default: 16)")
  parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help=" (default: 16)")
  parser.add_argument("--warmup_steps", type=int, default=500, help=" (default: 500)")
  parser.add_argument("--weight_decay", type=float, default=0.01, help=" (default: 0.01)")
  parser.add_argument("--logging_dir", type=str, default="./logs", help=" (default: ./logs)")
  parser.add_argument("--logging_steps", type=int, default=500, help=" (default: 100)")
  parser.add_argument("--evaluation_strategy", type=str, default="steps", help=" (default: steps)")
  parser.add_argument("--eval_steps", type=int, default=500, help=" (default: 500)")
  parser.add_argument("--load_best_model_at_end", type=bool, default=True, help=" (default: True)")
  parser.add_argument("--save_pretrained", type=str, default="./best_model", help=" (default: ./best_model)")
  parser.add_argument('--run_name', type=str, default="huggingface_test_3epoch")
  parser.add_argument("--n_splits", type=int, default=1, help=" (default: )")
  parser.add_argument("--test_size", type=float, default=0.1, help=" (default: )")
  parser.add_argument("--project_name", type=str, default="[Final] KoBART", help=" (default: )")
  parser.add_argument("--entity_name", type=str, default="growing_sesame", help=" (default: )")
  parser.add_argument("--report_to", type=str, default="wandb", help=" (default: )")
  parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help=" (default: )")
  


  args = parser.parse_args()
  print(args)

  seed_everything(args.seed)
  
  main(args)