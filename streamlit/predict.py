import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from utils import generate_next_token

def inference_from_kobart(model: nn.Module, input_str, store_name, customer_name):
    input_str = input_str.replace('\n', '')
    special_tokens = ['#@상호명#', '#@고객이름#', '#@위치#', '#@기관#', '#@전화번호#']

    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2", special_tokens=special_tokens)
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    input_ids = tokenizer.encode(input_str, return_tensors='pt')
    output = model.generate(input_ids, 
                            eos_token_id=1, 
                            max_length=128, 
                            num_beams=5, 
                            no_repeat_ngram_size=2, 
                            early_stopping=True)

    output = tokenizer.decode(output[0], skip_special_tokens=False)
    processed_output = postprocess(output, store_name=store_name, customer_name=customer_name)
    return processed_output


def inference_from_kogpt2(inf: nn.Module, input_str, store_name, customer_name):
    SUMMARY = '<unused1>'
    BOS = '</s>'
    EOS = '</s>'
    PTUNING = '<unused2>'

    input_str = input_str.replace('\n', '')
    tokenizer = inf.tokenizer
    input_tokens = tokenizer.encode(PTUNING)* 10 + tokenizer.encode(input_str) + tokenizer.encode(SUMMARY)
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

    processed_output = postprocess(predict, store_name=store_name, customer_name=customer_name)
    return processed_output


def postprocess(x, store_name, customer_name):
    output_x = x.replace("[a-zA-Z]+", "")
    output_x = output_x.replace("#@상호명#", store_name)
    output_x = output_x.replace("#@고객이름#", customer_name)
    output_x = output_x.replace("#@위치#", "")
    output_x = output_x.replace("<unk>", "")
    output_x = output_x.replace("</s>", "")
    output_x = output_x.replace("\n", " ")
    return output_x
