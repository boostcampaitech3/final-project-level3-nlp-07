import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm import tqdm
import pandas as pd

def main(args):
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

    original_df = pd.read_csv(args.path, encoding="utf-8")

    customer_review_list = original_df['고객리뷰'].tolist()
    manager_review_list = original_df['사장답글'].tolist()
    predicted_review_list = original_df['예측답글'].tolist()

    customer_emotion_list = []
    manager_emotion_list = []
    predicted_emotion_list = []

    print('Start Sentimental Analysis...')

    for outputs in pipe(customer_review_list):
        temp = set()

        for output in outputs:
            if output["score"]>0.4:
                temp.add(output["label"])

        customer_emotion_list.append(temp)

    for outputs in pipe(manager_review_list):
        temp = set()

        for output in outputs:
            if output["score"]>0.4:
                temp.add(output["label"])

        manager_emotion_list.append(temp)

    for outputs in pipe(predicted_review_list):
        temp = set()

        for output in outputs:
            if output["score"]>0.4:
                temp.add(output["label"])

        predicted_emotion_list.append(temp)
    
    print('Finish!')

    sentimental_manager = []
    sentimental_predicted = []

    for i in range(len(customer_emotion_list)):
        score_manager = len(customer_emotion_list[i] & manager_emotion_list[i]) / len(customer_emotion_list[i] | manager_emotion_list[i])
        sentimental_manager.append(score_manager)

        score_predicted = len(customer_emotion_list[i] & predicted_emotion_list[i]) / len(customer_emotion_list[i] | predicted_emotion_list[i])
        sentimental_predicted.append(score_predicted)

    print('Score Calculation Finished!')

    original_df['고객감정'] = customer_emotion_list
    original_df['사장감정'] = manager_emotion_list
    original_df['예측감정'] = predicted_emotion_list
    original_df['원본감정유사도'] = sentimental_manager
    original_df['예측감정유사도'] = sentimental_predicted

    original_df.to_csv('with_sentimental_' + args.path, index=False, encoding="utf-8")

    print(f'Saved as: with_sentimental_{args.path}')

    print(f'mean_original : {sum(sentimental_manager) / len(sentimental_manager)}')
    print(f'mean_predicted : {sum(sentimental_predicted) / len(sentimental_predicted)}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument("--path", type=str, default='kobart_beam.csv')
  args = parser.parse_args()
  print(args)

  main(args)