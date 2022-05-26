import pandas as pd
import torch
from torch.utils.data import Dataset
import re
from torch.utils.data import Dataset
import numpy as np

# class RE_Dataset(Dataset):
#     """ Dataset 구성을 위한 class."""
#     def __init__(self, pair_dataset, labels):
#         self.pair_dataset = pair_dataset
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
#         item['input_ids'] = self.pair_dataset['input_ids']
#         item['attention_mask'] = self.pair_dataset['attention_mask']
#         item['labels'] = self.labels['input_ids']
        
#         item["labels"] = [[-100 if token == "<pad>" else token for token in labels] for labels in batch["labels"]]
    

#         return item

#     def __len__(self):
#         return len(self.labels)


# def preprocess(sent):
#     return re.sub("\n", "", sent)


# def preprocessing_dataset(dataset):
#     """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
#     customer_review = []
#     store_review = []
#     menu = []

#     for m, c, s in zip(dataset['주문메뉴'], dataset['고객리뷰'], dataset['사장답글']):

#         # print("m:", m)
#         # print("c:", c)
#         # print("s:", s)
#         m = re.sub("\n", "", m).strip()
#         c = re.sub("\n", "", c).strip()
#         s = re.sub("\n", "", s).strip()

#         customer_review.append(m)
#         store_review.append(c)
#         menu.append(s)

#     out_dataset = pd.DataFrame({'menu': menu,'customer_review':customer_review,
#                                 'store_review':store_review,})

#     return out_dataset

# def load_data(dataset_dir):
#     """ csv 파일을 경로에 맡게 불러 옵니다. """
#     pd_dataset = pd.read_csv(dataset_dir, encoding='utf-8')
#     dataset = preprocessing_dataset(pd_dataset)

#     return dataset


# def tokenized_dataset(dataset, tokenizer):
#     """ tokenizer에 따라 sentence를 tokenizing 합니다."""
#     tokenized_input = tokenizer(
#         dataset['menu'].tolist(),
#         dataset['customer_review'].tolist(),
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512,
#     ) 

#     tokenized_output = tokenizer(
#         dataset['store_review'].tolist(),
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512,
#     ) 


#     return tokenized_input, tokenized_output


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = dataset
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(instance['고객리뷰'])
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['사장답글'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return self.len
    
    
class CustomTestDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = dataset
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(instance['고객리뷰'])
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['사장답글'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_padding_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return self.len