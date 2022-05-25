from os import listdir
import os
import numpy as np
import pandas as pd
import re

def concat_cafelist_files(DATA_DIR = "../data/", prefix_fname="total", output_fname="nonpre_total_cafes.csv"):
    files = [file for file in listdir(DATA_DIR) if file.startswith(prefix_fname)]
    
    total = pd.read_csv(os.path.join(DATA_DIR, files[0])).to_numpy()
    for x in files[1:]:
        total = np.vstack((total, pd.read_csv(os.path.join(DATA_DIR, x)).to_numpy()))
    total = pd.DataFrame(total, columns=['업체명', '맛','양','배달','주문메뉴','고객리뷰','사장답글'])
    total.to_csv(output_fname, encoding='utf-8')


def preprocessing_cafelist(input_fname="nonpre_total_cafes.csv", output_fname="pre_total_cafes.csv"):
    df = pd.read_csv(input_fname)
    
    print(len(df['사장답글'].unique()))
    
    del_newline = lambda x: x.str.replace(r'[\\n\n]', ' ', regex=True) # \n 제거
    del_space = lambda x: x.str.replace(r'[\s\\n]{2,}', ' ', regex=True) # \n 여러 개 -> 1개 ' '

    masking_username = lambda x: x.str.replace(r'[ㄱ-ㅎ가-힣\w]{0,}(\*\*){0,}(님|고객님)', '[고객이름]님', regex=True) # 영어**님 -> [고객 이름]님
    masking_cafename = lambda x, cafe: re.sub(f"{cafe}|{' '.join(cafe.split('-'))}|{'|'.join(cafe.split('-'))}", '[상호명]', x) # 카페이름 -> [상호명]
    
    df['주문메뉴'] = del_space(del_newline(df['주문메뉴']))
    df['고객리뷰'] = del_space(del_newline(df['고객리뷰']))
    df['사장답글'] = masking_username(del_space(del_newline(df['사장답글'])))
    df['사장답글'] = list(map(masking_cafename, df['사장답글'], df['업체명']))
    
    df.to_csv(output_fname, encoding='utf-8')
    print('Pre-processing DONE!')
