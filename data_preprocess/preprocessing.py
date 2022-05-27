from os import listdir
import os
import numpy as np
import pandas as pd
import re

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
    df['사장답글'] = list(map(masking_username, del_space(df['사장답글']), df['고객ID']))
    df['사장답글'] = list(map(masking_cafename, df['사장답글'], df['업체명']))
    
    df.to_csv(output_fname, encoding='utf-8')
    print('[DONE] Pre-processing!')
