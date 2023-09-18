from transformers import DistilBertForMaskedLM, DistilBertTokenizerFast
from transformers import PretrainedConfig
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_bert_ft = DistilBertForMaskedLM.from_pretrained("./model_dbertft/", from_tf=True)
tokenizer = DistilBertTokenizerFast.from_pretrained("bert-base-uncased")

def get_news_embeddings(news):
    #Gets the classifier layer (logits) of the model
    #Mean of the embeddings on the max length
    if not(isinstance(news, str)):
        news_text = news.text
    else:
        news_text = news
    text_tokens = tokenizer.encode(news_text, return_tensors="pt")
    model_input = 512
    embeddings = [model_bert_ft(text_tokens[:, :min(model_input, text_tokens.shape[1])])[0][:, :, :].detach().numpy()]
    if text_tokens.shape[1] > model_input:
        for i in range(1, text_tokens.shape[1] // model_input):
            embeddings.append(model_bert_ft(text_tokens[:, i * model_input : (i+1)*model_input])[0][:, :, :].detach().numpy())
        if (text_tokens.shape[1] % model_input) >= 1:
            final_embeddings = model_bert_ft(text_tokens[:, -(text_tokens.shape[1] % model_input) :])[0][:, :, :].detach().numpy()
            #pad the final embeddings to 512
            final_embeddings = np.concatenate((final_embeddings, np.zeros((1, 512 - final_embeddings.shape[1], final_embeddings.shape[-1]))), axis=1)
            embeddings.append(final_embeddings)
    return np.mean(embeddings, axis=0)

def get_cosine_similarity(a, b):
    return np.mean(cosine_similarity(a, b))

def get_user_embeddings(historical_news, current_news, historical_embeddings=None):
    historical_concatenated = ""
    for news in historical_news:
        try:
            historical_concatenated += news.text
        except:
            historical_concatenated += news
    if historical_embeddings is None:
        historical_embeddings = get_news_embeddings(historical_concatenated)
    try:
        current_embeddings = get_news_embeddings(current_news.text)
    except:
        current_embeddings = get_news_embeddings(current_news)
    return historical_embeddings, current_embeddings

def gmrf(y, alpha, beta):
    if beta:
        if alpha < 0.5:
            y = y**(2*alpha) + 1
        else:
            y = -y**(1/(2*(abs(alpha-1)+0.5)-1)) + 1
    else:
        if alpha < 0.5:
            y = (1-y)**(1/(2*alpha))
        else:
            y = (1-y)**(2*(abs(alpha-1)+0.5)-1)
    return y

def reswhy(rankings, rank_as_scores, weights, alphas, betas):
    n_articles = len(rankings[0][0])
    combined_scores = {i: 0 for i in rankings[0][0]}
    for p in range(len(rankings)):
        rank_as_score, weight, alpha, beta = rank_as_scores[p], weights[p], alphas[p], betas[p]
        ids, scores = rankings[p]
        if rank_as_score:
            scores = [i/n_articles for i in range(n_articles)]
        # Make sure that the rankings are already in the right order
        sort_order = np.argsort(scores)[::-1]
        scores, ids = np.array(scores)[sort_order], np.array(ids)[sort_order]
        if scores[0] < scores[-1]:
            scores = scores[::-1]
        min_score, max_score = scores[-1], scores[0]
        scores = [(score - min_score) / (max_score - min_score) for score in scores]
        scores = gmrf(np.array(scores), alpha, beta)
        for i in range(n_articles):
            combined_scores[ids[i]] += weight * scores[i]
    ids_final = np.array(list(combined_scores.keys()))[np.argsort(list(combined_scores.values()))[::-1]]
    return ids_final