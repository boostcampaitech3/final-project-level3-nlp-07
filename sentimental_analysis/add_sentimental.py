from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm import tqdm
import pandas as pd


model_name = "searle-j/kote_for_easygoing_people"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0, # gpu number, -1 if cpu used
        return_all_scores=True,
        function_to_apply='sigmoid'
    )

original_df = pd.read_csv('../final_total_data.csv', encoding="utf-8-sig")

customer_review_list = original_df['고객리뷰'].tolist()
manager_review_list = original_df['사장답글'].tolist()

customer_emotion_list = []

for outputs in tqdm(pipe(customer_review_list)):
    temp = set()

    for output in outputs:
        if output["score"]>0.4:
            temp.add(output["label"])

    customer_emotion_list.append(temp)

customer_emotion_list

manager_emotion_list = []

for outputs in tqdm(pipe(manager_review_list)):
    temp = set()

    for output in outputs:
        if output["score"]>0.4:
            temp.add(output["label"])

    manager_emotion_list.append(temp)

manager_emotion_list

emotion_intersection_list = []

for i, j in tqdm(zip(customer_emotion_list, manager_emotion_list)):
    emotion_intersection_list.append(set(i & j))


original_df['고객감정'] = customer_emotion_list
original_df['사장감정'] = manager_emotion_list
original_df['공통감정'] = emotion_intersection_list


original_df.to_csv("final_review_data_with_emotion_v1.csv", index=False, encoding="utf-8-sig")

