from pymongo import MongoClient
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
import numpy as np
import re
import joblib

client = MongoClient('localhost')
 
 
db = client["twitternewsrec"]
news_collection = db["news"]
users_collection = db["users"]
news_count = news_collection.count_documents({})
users_count = users_collection.count_documents({})
 
dico=joblib.load("C:\\Users\\natha\\CentraleSupelec\\Cours_CS\\3A\\Cours\\Filiere\\Periode_1\\Etude_de_cas\\Twinews-adapted\\twinews\\models\\test_batches.pkl")

class News():
    def __init__(self,url,text,users):
        self.url = url
        self.text = text
        self.users = users
    def __repr__(self) -> str:
        return(f"News object [url: {self.url}, text: {self.text[:100]}..., users: {self.users if self.users else None}]")
    
def get_news_list():
    news_list = []
    cursor = news_collection.find()
    print("Loading news from mongodb...")
    with tqdm(total=news_count) as pbar:
        while cursor.alive:
            for doc in cursor:
                pbar.update(1)
                news_list.append(News(doc['url'],doc['scrap']['text'],doc['users'] if 'users' in doc else None))    
    return news_list
 
class Users():
    TRAIN_TEST_RATIO = 0.8
    def __init__(self,user_id,news_url):
        self.user_id = user_id
        self.news_url = news_url
        self._get_news_text()
        n = len(self.news_text)
        self.train_list = self.news_text[:int(n*self.TRAIN_TEST_RATIO)]
        self.test_list=self.news_text[int(n*self.TRAIN_TEST_RATIO):]
 
    def _get_news_text(self):
        self.news_text = [None for _ in self.news_url]
        for i,url in enumerate(self.news_url):
            news=news_collection.find_one({"url":url})
            if(news is None):
                self.news_text[i] = None
                continue
            
            self.news_text[i] = news['scrap']['text']
 
        self.news_url = [news for i,news in enumerate(self.news_url) if self.news_text[i] is not None]
        self.news_text = [text for text in self.news_text if text is not None]
 
    def get_test_batch(self,n_batch = 100,read_ratio = 0.1):
        n_read = min(int(n_batch * 0.1),len(self.test_list))
        n_unread = n_batch - n_read

        test_list_text = list(np.random.choice(self.test_list,n_read,replace=False))
        test_list_isread = [1 for _ in range(n_read)]

        query = {"scrap.text": {"$nin": self.news_text}}  
        result = [news['scrap']['text'] for news in (news_collection.find(query))]
        chosen_result_indices = list(np.random.choice(len(result),n_unread,replace=False))
        test_list_text += [result[i] for i in chosen_result_indices]
        test_list_isread += [0 for _ in range(n_unread)]
        return test_list_text,test_list_isread
        
        
 
        
def get_users_list():
    users_list = []
    cursor = users_collection.find()
    print("Loading users from mongodb...")
    with tqdm(total=users_count) as pbar:
        while cursor.alive:
            for user in cursor:
                pbar.update(1)
                new_user = Users(user["user_id"],user['news'])
                users_list.append(new_user)  
 
    return users_list


def lemmatize(liste_text):
    liste_text=list(map(lambda x: re.sub(',','',x),liste_text))
    liste_text=list(map(lambda x: re.sub(';','',x),liste_text))
    liste_text=list(map(lambda x: re.sub('\.','',x),liste_text))
    liste_text=list(map(lambda x: re.sub('\\n','',x),liste_text))
    liste_text=list(map(lambda x: re.sub('\?','',x),liste_text))
    liste_text=list(map(lambda x: re.sub('\!','',x),liste_text))
    return list(map(lambda x: x.split(" "),liste_text))




users = get_users_list()  
dico_score={}
p=0
for h in range(len(users)):
    training=users[h].train_list
    training_corpus=list(map(lambda x: x.split(" "),training))
    training_corpus=lemmatize(training)
    corpus = training_corpus
    dictionary = Dictionary(corpus)
    bm25_model = OkapiBM25Model(dictionary=dictionary)
    bm25_corpus = bm25_model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),normalize_queries=False, normalize_documents=False)
    testing=dico[users[h].user_id]["news_texts"]
    relevant=dico[users[h].user_id]["news_is_relevant"]
    news_id=dico[users[h].user_id]["key"]
    testing_corpus=lemmatize(testing)
    testing_scores=[]
    for i in range(len(testing_corpus)):
        query = testing_corpus[i]
        tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')  # Enforce binary weighting of queries
        tfidf_query = tfidf_model[dictionary.doc2bow(query)]

        similarities = bm25_index[tfidf_query]
        testing_scores.append((news_id[i],sum(similarities),relevant[i]))
        dico_score[users[h].user_id]=testing_scores
        

try :
    joblib.dump(dico_score,"C:\\Users\\natha\\CentraleSupelec\\Cours_CS\\3A\\Cours\\Filiere\\Periode_1\\Etude_de_cas\\Twinews-adapted\\twinews\\models\\ranking.pkl")
except:
    print("issue with saving")