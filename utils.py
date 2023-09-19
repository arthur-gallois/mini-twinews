from pymongo import MongoClient  # Importer la bibliothèque MongoDB pour la connexion à la base de données
from tqdm import tqdm  # Importer la bibliothèque pour afficher des barres de progression

import numpy as np  # Importer la bibliothèque NumPy pour le traitement de données

# Établir une connexion à la base de données MongoDB (localhost par défaut)
client = MongoClient('localhost')

# Sélectionner la base de données "twitternewsrec"
db = client["twitternewsrec"]

# Sélectionner les collections "news" et "users" dans la base de données
news_collection = db["news"]
users_collection = db["users"]

# Compter le nombre de documents dans les collections "news" et "users"
news_count = news_collection.count_documents({})
users_count = users_collection.count_documents({})

# Définir une classe "News" pour représenter les données des actualités
class News():
    def __init__(self, url, text, users):
        self.url = url
        self.text = text
        self.users = users
    
    def __repr__(self) -> str:
        return f"News object [url: {self.url}, text: {self.text[:100]}..., users: {self.users if self.users else None}]"

# Fonction pour obtenir une liste d'actualités à partir de la base de données
def get_news_list():
    news_list = []
    cursor = news_collection.find()
    print("Loading news from MongoDB...")
    
    # Afficher une barre de progression basée sur le nombre de documents
    with tqdm(total=news_count) as pbar:
        while cursor.alive:
            for doc in cursor:
                pbar.update(1)
                news_list.append(News(doc['url'], doc['scrap']['text'], doc['users'] if 'users' in doc else None))
    
    return news_list

# Définir une classe "Users" pour représenter les données des utilisateurs
class Users():
    TRAIN_TEST_RATIO = 0.8
    
    def __init__(self, user_id, news_url):
        self.user_id = user_id
        self.news_url = news_url
        self._get_news_text()
        n = len(self.news_text)
        self.train_list = self.news_text[:int(n * self.TRAIN_TEST_RATIO)]
        self.test_list = self.news_text[int(n * self.TRAIN_TEST_RATIO):]

    def _get_news_text(self):
        self.news_text = [None for _ in self.news_url]
        for i, url in enumerate(self.news_url):
            news = news_collection.find_one({"url": url})
            if news is None:
                self.news_text[i] = None
                continue
            
            self.news_text[i] = news['scrap']['text']

        self.news_url = [news for i, news in enumerate(self.news_url) if self.news_text[i] is not None]
        self.news_text = [text for text in self.news_text if text is not None]

    def get_test_batch(self, n_batch=100, read_ratio=0.1):
        """
        Cette méthode génère un lot de test pour un utilisateur, composé d'une combinaison d'actualités lues et non lues.
        
        :param n_batch: Le nombre total d'actualités à inclure dans le lot de test.
        :param read_ratio: Le ratio d'actualités lues dans le lot de test (valeur entre 0 et 1).
        
        :return: Deux listes - la première contient le texte des actualités du lot de test, la seconde indique si chaque actualité est lue (1) ou non lue (0).
        """
        n_read = min(int(n_batch * read_ratio), len(self.test_list))
        n_unread = n_batch - n_read

        test_list_text = list(np.random.choice(self.test_list, n_read, replace=False))
        test_list_isread = [1 for _ in range(n_read)]

        query = {"scrap.text": {"$nin": self.news_text}}  
        result = [news['scrap']['text'] for news in (news_collection.find(query))]
        chosen_result_indices = list(np.random.choice(len(result), n_unread, replace=False))
        test_list_text += [result[i] for i in chosen_result_indices]
        test_list_isread += [0 for _ in range(n_unread)]
        return test_list_text, test_list_isread

# Fonction pour obtenir une liste d'utilisateurs à partir de la base de données
def get_users_list():
    users_list = []
    cursor = users_collection.find()
    print("Loading users from MongoDB...")
    
    # Afficher une barre de progression basée sur le nombre de documents
    with tqdm(total=users_count) as pbar:
        while cursor.alive:
            for user in cursor:
                pbar.update(1)
                new_user = Users(user["user_id"], user['news'])
                users_list.append(new_user)
    
    return users_list

# Obtenir la liste d'utilisateurs à partir de la base de données
users = get_users_list()
