import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from konlpy.tag import Okt
from konlpy.tag import Mecab
from sentence_transformers import SentenceTransformer, util

sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
mecab = Mecab()
morph = Okt()
okt = Okt()
filename = 'glove.6B.100d.txt.word2vec'
glove_model = KeyedVectors.load_word2vec_format(filename, binary=False)

def mecab_tokenizer(text):
    return mecab.morphs(text)

paths = ["predict_32_256/predict_", "predict_32_256_special/predict_"]

for path in paths:
    df = pd.read_csv(path)
    print("****" + path + " 시작******")

    # TF-IDF
    total_sim_tf = []
    tfidf_vectorizer = TfidfVectorizer(tokenizer=mecab_tokenizer)
    for fir, sec in zip(df['고객리뷰'], df['예측답글']):
        try:
            sentences = (fir, sec)
            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
            cos_similar = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            total_sim_tf.append(cos_similar[0][0])
        except Exception as e:
            total_sim_tf.append(0)
    df['tf-idf(고객-예측)'] = total_sim_tf

    total_sim_tf = []
    tfidf_vectorizer = TfidfVectorizer(tokenizer=mecab_tokenizer)
    for fir, sec in zip(df['사장답글'], df['예측답글']):
        try:
            sentences = (fir, sec)
            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
            cos_similar = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            total_sim_tf.append(cos_similar[0][0])
        except Exception as e:
            total_sim_tf.append(0)
    df['tf-idf(사장-예측)'] = total_sim_tf
    
    # Glove
    total_sim_glove = []
    for fir, sec in zip(df['고객리뷰'], df['예측답글']):
        try:
            f_n = okt.nouns(fir)
            s_n = okt.nouns(sec)
            f_total = []
            for f_w in f_n:
                try:
                    f_total.append(glove_model[f_w])
                except Exception as e:
                    continue
            f_emb = sum(f_total)/len(f_total)
            
            s_total = []
            for s_w in s_n:
                try:
                    s_total.append(glove_model[s_w])
                except Exception as e:
                    continue
            s_emb = sum(s_total)/len(s_total)
            res = cosine_similarity(f_emb.reshape(1,-1), s_emb.reshape(1,-1))
            total_sim_glove.append(res[0][0])
        except Exception as e:
            total_sim_glove.append(0)
    df['glove(고객-예측)'] = total_sim_glove

    total_sim_glove = []
    for fir, sec in zip(df['사장답글'], df['예측답글']):
        try:
            f_n = okt.nouns(fir)
            s_n = okt.nouns(sec)
            f_total = []
            for f_w in f_n:
                try:
                    f_total.append(glove_model[f_w])
                except Exception as e:
                    continue
            f_emb = sum(f_total)/len(f_total)
            
            s_total = []
            for s_w in s_n:
                try:
                    s_total.append(glove_model[s_w])
                except Exception as e:
                    continue
            s_emb = sum(s_total)/len(s_total)
            res = cosine_similarity(f_emb.reshape(1,-1), s_emb.reshape(1,-1))
            total_sim_glove.append(res[0][0])
        except Exception as e:
            total_sim_glove.append(0)
    df['glove(사장-예측)'] = total_sim_glove
    
    # SBERT
    total_sim_SBERT = []
    for fir, sec in zip(df['고객리뷰'], df['예측답글']):
        try:
            emb1 = sbert_model.encode(fir, convert_to_tensor=True)
            emb2 = sbert_model.encode(sec, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(emb1, emb2)
            total_sim_SBERT.append(float(cosine_scores[0][0]))
        except Exception as e:
            total_sim_SBERT.append(0)
    df['SBERT(고객-예측)'] = total_sim_SBERT

    total_sim_SBERT = []
    for fir, sec in zip(df['사장답글'], df['예측답글']):
        try:
            emb1 = sbert_model.encode(fir, convert_to_tensor=True)
            emb2 = sbert_model.encode(sec, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(emb1, emb2)
            total_sim_SBERT.append(float(cosine_scores[0][0]))
        except Exception as e:
            total_sim_SBERT.append(0)
    df['SBERT(사장-예측)'] = total_sim_SBERT
    df.to_csv(path)
