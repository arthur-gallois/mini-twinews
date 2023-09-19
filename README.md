# mini-twinews
Une implémentation simple et basique de la thèse de Julien Hay: Apprentissage de la représentation du style écrit, application à la recommandation d’articles d’actualité. La thèse est accessible à l'adresse <https://theses.hal.science/tel-03420487>.

Ce projet ne contient pas de dataset, pour obtenir un dataset, utilisez cet outil. <https://github.com/hayj/Twinews>.

Ce projet possède le code necessaire pour classer des articles d'un base de donnée à partir de l'historique d'un utilisateur grâce aux algorithmes bm25 et d-bert-ft.

## Importation de la base de donnée

Les fichiers de la base de données users et news doivent être placés à la racine du projet. Mongodb doit être installé et lancé sur l'ordinateur en local.
Il faut ensuite lancer indexor.py pour remplir une base de donnée nommée twitternewsrec à partir des archives contenues dans users et news.

## Utilisation de la base de données

Le fichier utils.py contient une base de code d'exploitation de la base de donnée.

    get_users_list() # Renvoie une lists d'objets users
    get_news_list()  # Renvoie une liste d'objets news

Les objets news sont des objets contenant les propriétés suivantes:

    url: L'url de la news
    text: Le texte de la news
    users: La liste des user_id ayant retweeté la news (ou posté)

Les objets users sont des objets contenant les propriétés suivantes:

    user_id: l'identifiant de l'utilisateur (variable de la base de donnée)
    news_url: La liste des url de news postées
    train_list: Une partie de news_url destinée à l'entrainement (aussi appelé historique dans l'article)
    test_list: Une partie de news_url destinée à l'évaluation



