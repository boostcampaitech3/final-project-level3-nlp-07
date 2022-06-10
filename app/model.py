import yaml
import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel, BartForConditionalGeneration
from app.postprocess import postprocess
from app.train import KoGPTConditionalGeneration

special_tokens = ['#@상호명#', '#@고객이름#', '#@위치#', '#@기관#', '#@전화번호#']


class KoBART(nn.Module):
    def __init__(self, model_path: str = None):
        super().__init__()
        MODEL_NAME = "gogamza/kobart-base-v2"

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME,
                bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                pad_token='<pad>', mask_token='<mask>', special_tokens=special_tokens)
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.model = BartForConditionalGeneration.from_pretrained(model_path if model_path else MODEL_NAME)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_str, store_name, customer_name) -> torch.Tensor:
        input_str = input_str.replace('\n', '')

        input_ids = self.tokenizer.encode(input_str, return_tensors='pt')
        output = self.model.generate(input_ids, 
                                    eos_token_id=self.tokenizer.eos_token_id, 
                                    max_length=128, 
                                    num_beams=5, 
                                    no_repeat_ngram_size=2, 
                                    early_stopping=True,
                                    use_cache=True)

        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        processed_output = postprocess(output, store_name=store_name, customer_name=customer_name)
        return processed_output


class KoGPT2(nn.Module):
    def __init__(self, hparams_file: str = None, model_path: str = None):
        super().__init__()
        self.max_length_token = 256

        with open(hparams_file) as f:
            hparams = yaml.load(f)

        self.SUMMARY = '<unused1>'
        self.PTUNING = '<unused2>'
        self.EOS = '</s>'
        self.BOS = '</s>'

        self.inf = KoGPTConditionalGeneration.load_from_checkpoint(model_path, hparams=hparams)
        # self.inf.model.resize_token_embeddings(len(self.tokenizer))

        self.tokenizer = self.inf.tokenizer
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def forward(self, input_str, store_name, customer_name) -> torch.Tensor:
        input_str = input_str.replace('\n', '')

        try:
            input_tokens = self.tokenizer.encode(self.BOS) + self.tokenizer.encode(input_str) + self.tokenizer.encode(self.SUMMARY)
            input_tensor = torch.tensor(input_tokens).unsqueeze(0)
            output = self.inf.model.generate(input_ids=input_tensor, 
                                                max_length=self.max_length_token, 
                                                num_beams=5, 
                                                no_repeat_ngram_size=2, 
                                                early_stopping=True,
                                                use_cache=True)
            predict = self.tokenizer.decode(output[0], skip_special_tokens=False).split(self.SUMMARY)[-1].strip()
        except Exception as e:
            predict = "오류 발생"

        processed_output = postprocess(predict, store_name=store_name, customer_name=customer_name)
        return processed_output

