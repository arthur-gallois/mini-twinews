from pymongo import MongoClient
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
import numpy as np
import re
import joblib
from utils import *

dico=joblib.load("test_batches.pkl")    #load the test dataset

def lemmatize(liste_text):
    liste_text=list(map(lambda x: re.sub(',','',x.lower()),liste_text))
    liste_text=list(map(lambda x: re.sub(';','',x),liste_text))
    liste_text=list(map(lambda x: re.sub('\.','',x),liste_text))
    liste_text=list(map(lambda x: re.sub('\\n','',x),liste_text))
    liste_text=list(map(lambda x: re.sub('\?','',x),liste_text))
    liste_text=list(map(lambda x: re.sub('\!','',x),liste_text))
    return list(map(lambda x: x.split(" "),liste_text))


users = get_users_list()  
dico_score={}
for h in tqdm(range(len(users))):
    training=users[h].train_list
    corpus=lemmatize(training)
    dictionary = Dictionary(corpus)
    bm25_model = OkapiBM25Model(dictionary=dictionary,k1=2.4,b=1.2,)
    bm25_corpus = bm25_model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),normalize_queries=False, normalize_documents=False)
    testing=dico[users[h].user_id]["news_texts"]
    relevant=dico[users[h].user_id]["news_is_relevant"]
    news_id=dico[users[h].user_id]["key"]
    testing_corpus=lemmatize(testing)
    testing_scores=[]
    for i in range(len(testing_corpus)):
        query = testing_corpus[i]
        tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
        tfidf_query = tfidf_model[dictionary.doc2bow(query)]
        similarities = bm25_index[tfidf_query]
        testing_scores.append((news_id[i],sum(similarities),relevant[i]))
    dico_score[users[h].user_id]=testing_scores        

try :
    joblib.dump(dico_score,"ranking_BM25.pkl")
except:
    print("issue with saving")