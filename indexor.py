import os  
from pymongo import MongoClient 
import json 
import bz2  
from tqdm import tqdm  
import multiprocessing  
from joblib import Parallel, delayed  

# Obtenir le nombre de cœurs CPU disponibles sur la machine
num_cores = multiprocessing.cpu_count()

# Obtenir le chemin du répertoire racine
DIR_PATH = os.path.split(os.path.abspath(__file__))[0]

# Définir les chemins vers les répertoires contenant les données à importer
NEWS_PATH = os.path.join(DIR_PATH, "news")
USERS_PATH = os.path.join(DIR_PATH, "users")

# Établir une connexion à la base de données MongoDB (local par défaut)
client = MongoClient()
db = client["twitternewsrec"]
news_collection = db["news"]
users_collection = db["users"]

# Supprimer les collections existantes dans la base de données
news_collection.drop()
users_collection.drop()

# Créer des index pour accélérer les opérations de recherche
news_collection.create_index("lastUrlDomain", unique=False)
news_collection.create_index("url", unique=True)
users_collection.create_index("user_id", unique=True)

# Fonction pour importer des données JSON depuis un fichier
def import_news(filename):
    filepath = os.path.join(NEWS_PATH, filename)
    with bz2.open(filepath) as file:
        for line in file.readlines():
            try:
                data = json.loads(line)
            except:
                data = json.loads(line.decode('utf-8'))
            news_collection.insert_one(data)

# Fonction pour importer des données JSON d'utilisateurs depuis un fichier
def import_users(filename):
    filepath = os.path.join(USERS_PATH, filename)
    with bz2.open(filepath) as file:
        for line in file.readlines():
            try:
                data = json.loads(line)
            except:
                data = json.loads(line.decode('utf-8'))
            users_collection.insert_one(data)

print("Importing news...")
news_filenames = tqdm(os.listdir(NEWS_PATH))

# Importer les données des actualités en parallèle
processed_list = Parallel(n_jobs=num_cores, backend='threading')(delayed(import_news)(filename) for filename in news_filenames)

print("Importing users...")
users_filenames = tqdm(os.listdir(USERS_PATH))

# Importer les données d'utilisateurs en parallèle
processed_list = Parallel(n_jobs=num_cores, backend='threading')(delayed(import_users)(filename) for filename in users_filenames)

print("Mapping users to news...")
newsUsersMapping = dict()
users_id = tqdm(users_collection.distinct("user_id"))

# Fonction pour créer la correspondance entre les utilisateurs et les actualités
def setNewsUsersMapping(id):
    data = users_collection.find_one({"user_id": id})
    for news in data["news"]:
        if news not in newsUsersMapping:
            newsUsersMapping[news] = []
        newsUsersMapping[news].append(id)

# Créer la correspondance en parallèle
processed_list = Parallel(n_jobs=num_cores, backend='threading')(delayed(setNewsUsersMapping)(user_id) for user_id in users_id)

print("Inserting mapping in the database...")
mappings = tqdm(newsUsersMapping.items())

# Fonction pour insérer la correspondance entre les utilisateurs et les actualités dans la base de données
def insertNewsUsersMapping(mapping):
    url, ids = mapping
    news_collection.update_one({"url": url}, {"$set": {"users": ids}})

# Insérer la correspondance en parallèle
processed_list = Parallel(n_jobs=num_cores, backend='threading')(delayed(insertNewsUsersMapping)(mapping) for mapping in mappings)

print("Done!")
news_count = news_collection.count_documents({})
users_count = users_collection.count_documents({})
print(f"Number of news: {news_count} | Number of users: {users_count}")
