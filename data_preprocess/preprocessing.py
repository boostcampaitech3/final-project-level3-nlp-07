from os import listdir
import os
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from pororo import Pororo

def concat_cafelist_files(DATA_DIR = "../data/", prefix_fname="total", output_fname="nonpre_total_cafes.csv"):
    files = [file for file in listdir(DATA_DIR) if file.startswith(prefix_fname)]
    print(f'{prefix_fname} 로 시작하는 파일 목록:', files)

    total = pd.read_csv(os.path.join(DATA_DIR, files[0])).to_numpy()
    for x in files[1:]:
        total = np.vstack((total, pd.read_csv(os.path.join(DATA_DIR, x)).to_numpy()))
    total = pd.DataFrame(total, columns=['업체명', '맛','양','배달','주문메뉴','고객id','고객리뷰','사장답글'])
    total.to_csv(output_fname, encoding='utf-8')
    print('[DONE] Concate cafe_list files!')


def masking_username(x, uname): 
    uname = uname.replace("*", "\*")
    if uname == "   손님":
        return re.sub(r"[ㄱ-ㅎ가-힣\w]+(\*\*){0,} {0,}(님|고객님)", '#@고객이름#', x)
    return re.sub(f"{uname}|{uname[:-1]}|{uname[:-3]}", '#@고객이름#', x) # 영어**님 -> [고객 이름]님

def masking_cafename(x, cafe): 
    cafename = re.sub(f"{cafe}|{' '.join(cafe.split('-'))}|{'|'.join(cafe.split('-'))}", '#@상호명#', x)
    return re.sub(r'(#@상호명#){2,}', '#@상호명#', cafename) # 카페이름 -> [상호명]

def preprocessing_cafelist(input_fname="nonpre_total_cafes.csv", output_fname="pre_total_cafes.csv"):
    df = pd.read_csv(input_fname)
    
    print('unique한 사장 답글 개수:', len(df['사장답글'].unique()))
    
    del_space = lambda x: x.str.replace(r'[\s\\n]{2,}', '\n', regex=True) # \n 여러 개 -> 1개 \n
    
    df['주문메뉴'] = del_space(df['주문메뉴'])
    df['고객리뷰'] = del_space(df['고객리뷰'])
    df['사장답글'] = list(map(masking_username, del_space(df['사장답글']), df['고객id']))
    df['사장답글'] = list(map(masking_cafename, df['사장답글'], df['업체명']))
    df['사장답글'] = masking_org_n_loc_entity(df['사장답글'])

    df.to_csv(output_fname, encoding='utf-8')
    print('[DONE] Pre-processing!')


def masking_org_n_loc_entity(x):

    ner = Pororo(task="ner", lang="ko")
    ner_masking_map = {"ORGANIZATION": "#@기관#", "LOCATION": "#@위치#"}

    new_x = deque([])
    
    for single_x in tqdm(x):
        try:
            res = defaultdict(list)
            tmp = set([(word, ner_masking_map[ty]) for word, ty in ner(single_x) if ty in ["ORGANIZATION", "LOCATION"]])
            _= [res[v].append(k) for k,v in tmp  if '#' not in k and '@' not in k]

            new_single_x = single_x

            if "#@기관#" in res.keys():
                new_single_x = re.sub('|'.join(res["#@기관#"]), "#@기관#", new_single_x) 
            if "#@위치#" in res.keys():
                new_single_x = re.sub('|'.join(res["#@위치#"]), "#@위치#", new_single_x)
            new_x.append(new_single_x)

        except KeyError as e: # EMOJI 있으면 ERROR
            new_x.append(single_x)
    return new_x



if __name__ == "__main__":
    # python preprocessing.py
    csv_filename = "../final_total_data.csv"

    preprocessing_cafelist(csv_filename, "pre_"+csv_filename)
