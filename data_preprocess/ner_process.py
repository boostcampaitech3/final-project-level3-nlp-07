import pandas as pd
import argparse
from collections import defaultdict, deque
from tqdm import tqdm
from pororo import Pororo
import re


def masking_org_n_loc_entity(x):
    ner = Pororo(task="ner", lang="ko")
    ner_masking_map = {"ORGANIZATION": "#@기관#", "LOCATION": "#@위치#"}

    new_x = deque([])
    for sent in tqdm(x):
        try:
            suitable_entities = defaultdict(list)
            suitable_entity = set([(word, ner_masking_map[ty]) for word, ty in ner(sent) if ty in ["ORGANIZATION", "LOCATION"] and word not in '#' and word not in '@'])            
            _= [suitable_entities[v].append(k) for k,v in suitable_entity]
            
            new_sent = sent
            if "#@기관#" in suitable_entities.keys():
                new_sent = re.sub('|'.join(suitable_entities["#@기관#"]), "#@기관#", new_sent) 
            if "#@위치#" in suitable_entities.keys():
                new_sent = re.sub('|'.join(suitable_entities["#@위치#"]), "#@위치#", new_sent)
            new_x.append(new_sent)

        except:
            new_x.append(sent)
    return list(new_x)


def masking_ner_parallelly(total, div_n = 50, i = 0):
    n = len(total)
    unit = n // div_n + 1
    print(f'parallel 수행할 총 사장답글 개수 (div_n: {div_n}, i: {i}): {unit} /{n}')        
    
    ner_comments = []
    for i in range(div_n):
        ner_comments.append(masking_org_n_loc_entity(total[unit*i : unit*(i+1)]['사장답글']))

    res = []
    _= [res.extend(x) for x in ner_comments]
    print(f'parallel 처리된 총 사장답글 개수 : {len(res)}')

    return res


if __name__ == "__main__":
    # python data_preprocess/ner_process.py --div_n 80 --i 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--div_n', type=int, help='init block no')
    parser.add_argument('--i', type=int, help='init i-th block')
    args = parser.parse_args()

    csv_filename = "final_total_data.csv"
    df = pd.read_csv(csv_filename)

    comments_deque = masking_ner_parallelly(df, div_n=args.div_n)
    pd.Series(comments_deque).to_csv(f"pre_{csv_filename[:-4]}_{args.div_n}_{args.i}.csv", encoding='utf-8', index=False)
    print('DONE!')
