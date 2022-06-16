import yaml
import torch
from train import KoGPTConditionalGeneration
from utils import generate_next_token
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


seed_everything(42)
hparams_file = 'logs/tb_logs/default/version_0/hparams.yaml'
with open(hparams_file) as f:
    hparams = yaml.load(f)

df = pd.read_csv('test.csv')

paths = [
    # 'logs/model_chp/epoch=00.ckpt',
    # 'logs/model_chp/epoch=01.ckpt',
    # 'logs/model_chp/epoch=02.ckpt',
    # 'logs/model_chp/epoch=03.ckpt',
    'logs/model_chp/epoch=04.ckpt',
]
for idx, path in tqdm(enumerate(paths)):
    inf = KoGPTConditionalGeneration.load_from_checkpoint(path, hparams=hparams)
    tokenizer = inf.tokenizer
    SUMMARY = '<unused1>'
    PTUNING = '<unused2>'
    EOS = '</s>'
    BOS = '</s>'

    customs = []
    predictions = []
    labels = []
    cnt = 0
    for custom, owner in tqdm(zip(df['고객리뷰'], df['사장답글']), total=100):
        if cnt > 100:
            break
        cnt += 1
        customs.append(custom)
        labels.append(owner)
        text = custom
        try:
            text = text.replace('\n', '')
            # input_tokens = tokenizer.encode(PTUNING)* 10 + tokenizer.encode(text) + tokenizer.encode(SUMMARY)
            input_tokens = tokenizer.encode(BOS) + tokenizer.encode("맛: ") + tokenizer.encode(df['맛'].iloc[cnt]) + \
                           tokenizer.encode("양: ") + tokenizer.encode(df['양'].iloc[cnt]) +  tokenizer.encode("배달: ") + tokenizer.encode(df['배달'].iloc[cnt]) + \
                           tokenizer.encode("리뷰: ") + tokenizer.encode(df['고객리뷰'].iloc[cnt]) + tokenizer.encode(SUMMARY)
            input_tensor = torch.tensor(input_tokens).unsqueeze(0)
            eos_id = tokenizer.encode(EOS)[0]

            limit = 0
            while True:
                if limit > 200:
                    break
                limit += 1
                pred = inf.model(input_tensor)
                next_token = generate_next_token(pred.logits, temperature=1.0, top_p=0.8)

                if next_token.item() == eos_id:
                    break
                else:
                    input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)],1)
            predict = tokenizer.decode(input_tensor[0]).split('<unused1>')[-1].strip()
            # output = inf.model.generate(input_ids=input_tensor, max_length=256, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
            # output = inf.model.generate(input_ids=input_tensor, max_length=256)
            # predict = tokenizer.decode(output[0]).split('<unused1>')[-1].strip()
        except Exception as e:
            predict = "오류 발생"
        predictions.append(predict)
    df1 = pd.DataFrame({'고객리뷰':customs, '예측답글':predictions,'사장답글':labels})
    df1.to_csv('greedy_100_2.csv', index=False)
