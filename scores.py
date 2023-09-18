import numpy as np
import re
import joblib
from sklearn.metrics import ndcg_score

dico=joblib.load("C:\\Users\\natha\\CentraleSupelec\\Cours_CS\\3A\\Cours\\Filiere\\Periode_1\\Etude_de_cas\\Twinews-adapted\\twinews\\models\\ranking.pkl")

def nDCG(scores):
    k=100
    DCG,iDCG=0,0
    for i in range(k):
        DCG+=scores[i][1]/np.log2(i+2)
    j=10
    #j=sum(testing[1])
    for i in range(j):
        iDCG+=1/np.log2(i+2)
    return DCG/iDCG

def get_ndcg_reswhy(ranking_bm25, ranking_dbertft, alphas, betas, weight, CV=False, return_ranking=False):
    ranking_reswhy = {}
    rank_as_scores = [False, False]
    weights = [weight, 1 - weight]

    ndcg_bm25 = []
    ndcg_dbertft = []
    ndcg_reswhy = []

    for user in users_keys:
        bm25 = ([ranking_bm25[user][i][0] for i in range(len(ranking_bm25[user]))], [ranking_bm25[user][i][1] for i in range(len(ranking_bm25[user]))])
        dbertft = ([ranking_dbertft[user][i][0] for i in range(len(ranking_dbertft[user]))], [ranking_dbertft[user][i][1] for i in range(len(ranking_dbertft[user]))])
        ground_truth = {ranking_bm25[user][i][0]: ranking_bm25[user][i][-1] for i in range(len(ranking_bm25[user]))}
        rankings = [bm25, dbertft]
        reswhy_ranking = reswhy(rankings, rank_as_scores, weights, alphas, betas)
        ground_truth_ordered = np.array([ground_truth[i] for i in reswhy_ranking]).reshape(1, -1)
        keys = np.array([i for i in range(len(reswhy_ranking))]).reshape(1, -1)
        ndcg_bm25.append(ndcg_score(np.array(list(ground_truth.values())).reshape(1, -1), np.array(bm25[1]).reshape(1, -1)))
        ndcg_dbertft.append(ndcg_score(np.array(list(ground_truth.values())).reshape(1, -1), np.array(dbertft[1]).reshape(1, -1)))
        ndcg_reswhy.append(ndcg_score(ground_truth_ordered, keys))
        ranking_reswhy[user] = [(i, i, ground_truth_ordered[0][i]) for i in range(len(reswhy_ranking))]
    
    print("nDCG BM25: ", np.mean(ndcg_bm25))
    print("nDCG DBERTFT: ", np.mean(ndcg_dbertft))
    print("nDCG RESWHY: ", np.mean(ndcg_reswhy))
    
    if return_ranking:
        return ranking_reswhy

    if CV:
        #CV in a way by only selecting the mean on a subset of the users
        return np.mean(np.array(ndcg_reswhy)[np.random.choice(len(ndcg_reswhy), 25, replace=False)])
    else:
        return np.mean(ndcg_reswhy)

def grid_search_reswhy(ranking_bm25, ranking_dbertft):
    alphas = np.linspace(0.01, 0.99, 10)
    betas = [[True, False], [False, True], [True, True], [False, False]]
    weight = np.linspace(0, 1, 10)

    best_alpha1, best_alpha2, best_beta, best_weight = 0, 0, [True, False], 0
    ndcg_max = 0

    for alpha1 in alphas:
        for alpha2 in alphas:
            for beta in betas:
                for w in weight:
                    ndcg_current = get_ndcg_reswhy(ranking_bm25, ranking_dbertft, [alpha1, alpha2], beta, w, CV=True)
                    if ndcg_current > ndcg_max:
                        ndcg_max = ndcg_current
                        best_alpha1, best_alpha2, best_beta, best_weight = alpha1, alpha2, beta, w

    best_ndcg = get_ndcg_reswhy(ranking_bm25, ranking_dbertft, [best_alpha1, best_alpha2], best_beta, best_weight, CV=False)
    print(f"Best alpha1: {best_alpha1}, best alpha2: {best_alpha2}, best beta: {best_beta}, best weight: {best_weight}, best ndcg: {best_ndcg}")

    return {"alpha1": best_alpha1, "alpha2": best_alpha2, "beta": best_beta, "weight": best_weight, "ndcg": best_ndcg}