import os
from pymongo import MongoClient
import json
import bz2
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

DIR_PATH = os.path.split(os.path.abspath(__file__))[0]
NEWS_PATH = os.path.join(DIR_PATH,"news")
USERS_PATH = os.path.join(DIR_PATH,"users")

client = MongoClient()
db = client["twitternewsrec"]
news_collection = db["news"]

users_collection = db["users"]

news_collection.drop()
users_collection.drop()

news_collection.create_index("lastUrlDomain",unique = False)
news_collection.create_index("url",unique = True)
users_collection.create_index("user_id",unique = True)



def import_news(filename):
    filepath = os.path.join(NEWS_PATH,filename)
    with bz2.open(filepath) as file:
        for line in file.readlines():
            try:
                data = json.loads(line)
            except:
                data = json.loads(line.decode('utf-8'))
            news_collection.insert_one(data)

def import_users(filename):
        filepath = os.path.join(USERS_PATH,filename)
        with bz2.open(filepath) as file:
            for line in file.readlines():
                try:
                    data = json.loads(line)
                except:
                    data = json.loads(line.decode('utf-8'))
                users_collection.insert_one(data)

print("Importing news...")
news_filenames = tqdm(os.listdir(NEWS_PATH))
processed_list = Parallel(n_jobs=num_cores,backend='threading')(delayed(import_news)(filename) for filename in news_filenames)

print("Importing users...")
users_filenames = tqdm(os.listdir(USERS_PATH))
processed_list = Parallel(n_jobs=num_cores,backend='threading')(delayed(import_users)(filename) for filename in users_filenames)



print("Mapping users to news...")
newsUsersMapping = dict()
users_id = tqdm(users_collection.distinct("user_id"))
def setNewsUsersMapping(id):
    data = users_collection.find_one({"user_id": id})
    for news in data["news"]:
        if news not in newsUsersMapping:
            newsUsersMapping[news] = []
        newsUsersMapping[news].append(id)

processed_list = Parallel(n_jobs=num_cores,backend='threading')(delayed(setNewsUsersMapping)(user_id) for user_id in users_id)    

print("Inserting mapping in the database...")
mappings = tqdm(newsUsersMapping.items())
def insertNewsUsersMapping(mapping):
    url,ids = mapping
    news_collection.update_one({"url": url}, {"$set": {"users": ids}})
     
processed_list = Parallel(n_jobs=num_cores,backend='threading')(delayed(insertNewsUsersMapping)(mapping) for mapping in mappings)    
    
print("Done !")
news_count = news_collection.count_documents({})
users_count = users_collection.count_documents({})
print(f"Number of news: {news_count} | Number of users: {users_count}")