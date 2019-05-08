import numpy as np
import scipy.sparse
import pickle
import json
import itertools
import re
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

with open("./app/irsystem/controllers/TVTropesScraper/Film/Film_tropes_dataset3.json", 'r') as f:
    movie_tropes_data = json.load(f)
with open("./app/irsystem/controllers/TVTropesScraper/Literature/Literature_tropes_dataset3.json", 'r') as f:
    book_tropes_data = json.load(f)
    
movies = movie_tropes_data.keys()
books = book_tropes_data.keys()

sparse_mbt = scipy.sparse.load_npz("./app/irsystem/controllers/SPARSE OR NECESSARY/sparse_mbt.npz")
sparse_bbt = scipy.sparse.load_npz("./app/irsystem/controllers/SPARSE OR NECESSARY/sparse_bbt.npz")
movie_popularities = pickle.load(open("./app/irsystem/controllers/SPARSE OR NECESSARY/movie_popularities.p", "rb" ))
book_popularities = pickle.load(open("./app/irsystem/controllers/SPARSE OR NECESSARY/book_popularities.p", "rb" ))
common_tropes = pickle.load(open("./app/irsystem/controllers/SPARSE OR NECESSARY/common_tropes.p", "rb" ))
col_to_trope_list = pickle.load(open("./app/irsystem/controllers/SPARSE OR NECESSARY/col_to_trope_list.p", "rb" ))
movie_titles = pickle.load(open("./app/irsystem/controllers/SPARSE OR NECESSARY/movie_titles.p", "rb" ))
book_titles = pickle.load(open("./app/irsystem/controllers/SPARSE OR NECESSARY/book_titles.p", "rb" ))

with open('./app/irsystem/controllers/SPARSE OR NECESSARY/book_word_to_trope.json', 'r') as f: 
    book_word_to_trope = json.load(f)
with open('./app/irsystem/controllers/SPARSE OR NECESSARY/movie_word_to_trope.json', 'r') as f: 
    movie_word_to_trope = json.load(f)
book_to_movie_vectorizer = pickle.load(open("./app/irsystem/controllers/SPARSE OR NECESSARY/book_to_movie_vectorizer.pickle", "rb" ))
movie_to_book_vectorizer = pickle.load(open("./app/irsystem/controllers/SPARSE OR NECESSARY/movie_to_book_vectorizer.pickle", "rb" ))
movie_tf_idf = pickle.load(open("./app/irsystem/controllers/SPARSE OR NECESSARY/movie_tf_idf.pickle", "rb" ))
book_tf_idf = pickle.load(open("./app/irsystem/controllers/SPARSE OR NECESSARY/book_tf_idf.pickle", "rb" ))
model = KeyedVectors.load("./app/irsystem/controllers/SPARSE OR NECESSARY/tbwb_model.bin")


def get_closest_tropes_to_keyword(keyword, word_to_trope, model, top_k = 5): 
    if keyword in model.vocab: 
        all_words = list(word_to_trope.keys())
        all_words = [word for word in all_words if word in model.vocab]
        dists = model.distances(keyword, all_words)
        sorted_indices = np.argsort(dists)
        sorted_keyword_match = [all_words[idx] for idx in sorted_indices[:top_k]]
        for word in sorted_keyword_match: 
            trope_matches = list(itertools.chain.from_iterable([word_to_trope[word] for word in sorted_keyword_match if word in word_to_trope]))
        return trope_matches[:top_k]
    else: 
        print('`{}` not in model vocabulary, cannot enhance search with keyword'.format(keyword))
        return []

def best_titles_by_tropes_enhanced(title, from_dataset, keyword, word_to_trope, vectorizer, to_tf_idf_matrix, model): 
    # documentation in jupyter notebook
    title_tropes =  from_dataset[title]    
    top_k_tropes = int(len(title_tropes)/2)
    most_similar_tropes = get_closest_tropes_to_keyword(keyword, word_to_trope, model, top_k_tropes)
    query_tropes = most_similar_tropes + title_tropes 
    query_vector = vectorizer.transform([' '.join(query_tropes)])
    similarity_scores = cosine_similarity(query_vector, to_tf_idf_matrix).flatten()
    return similarity_scores

def search_by_keyword(title, keyword, direction):
    if direction == 'bm':
        from_dataset = book_tropes_data
        to_dataset = movie_tropes_data
        word_to_trope = movie_word_to_trope
        vectorizer = book_to_movie_vectorizer
        to_tf_idf_matrix = movie_tf_idf
    elif direction == 'mb': # movie to book
        from_dataset = movie_tropes_data
        to_dataset = book_tropes_data
        word_to_trope = book_word_to_trope
        vectorizer = movie_to_book_vectorizer
        to_tf_idf_matrix = book_tf_idf
    else:
        raise Exception("Bad direction")
    
    similarity_scores = best_titles_by_tropes_enhanced(title, from_dataset, keyword, word_to_trope, vectorizer, to_tf_idf_matrix, model)
    return similarity_scores

def top_tropes_from_vector(v, n_tropes, col_to_trope_list):
    top_dot = np.argsort(-v)
    top_tropes = []
    for i in top_dot[:n_tropes]:
        if v[i] != 0:
            top_tropes.append(col_to_trope_list[i])
    return top_tropes

def add_rocchio(q_vec, titles, direction, beta=0.3):
    if direction == 'mb':
        doc_titles = book_titles
        doc_by_trope = sparse_bbt
    else:
        doc_titles = movie_titles
        doc_by_trope = sparse_mbt
    for title in titles:
        i = doc_titles.index(title)
        q_vec = q_vec + beta * doc_by_trope[i]
    return q_vec
        

def find_relevant(title, keyword = None, n_recs = 5, n_tropes=5, direction='mb', popularity_weight=0, keyword_weight=5, rocchio_titles = None):
    if popularity_weight is None: popularity_weight = 0
    popularity_weight = float(popularity_weight)
    
    if direction=='mb':
        if title not in movie_titles: return False
        i = movie_titles.index(title)
        query_vec = sparse_mbt[i]
        if rocchio_titles is not None:
            query_vec = add_rocchio(query_vec, rocchio_titles, direction=direction, beta=0.3)
        similarities = np.ndarray.flatten((sparse_bbt @ query_vec.T).A)
        if keyword is not None:
            keyword_vec = search_by_keyword(title, keyword, direction)
            similarities += keyword_vec*keyword_weight
        if popularity_weight > 0:
            similarities = np.multiply(similarities, popularity_weight * book_popularities)
        sorted_titles = np.flip(np.argsort(similarities), axis=0)
        recs, scores, top_tropes = [], [], []

        i = -1
        while len(recs) < 5:
            try:
                i += 1
                dot = np.ndarray.flatten(sparse_bbt[sorted_titles[i]].multiply(query_vec).A)
                top_tropes_of_result = top_tropes_from_vector(dot, n_tropes, col_to_trope_list)
                if len(top_tropes_of_result) < 5: continue     # enforce top 5 tropes
                recs.append(book_titles[sorted_titles[i]])
                scores.append(similarities[sorted_titles[i]])
                top_tropes.append(top_tropes_of_result)
            except IndexError:
                break

    elif direction=='bm':
        if title not in book_titles: return False
        i = book_titles.index(title)
        query_vec = sparse_bbt[i]
        if rocchio_titles is not None:
            query_vec = add_rocchio(query_vec, rocchio_titles, direction=direction, beta=0.3)
        similarities = np.ndarray.flatten((sparse_mbt @ query_vec.T).A)
        if keyword is not None:
            keyword_vec = search_by_keyword(title, keyword, direction)
            similarities += keyword_vec*keyword_weight
        if popularity_weight > 0:
            similarities = np.multiply(similarities, popularity_weight * movie_popularities)
        sorted_titles = np.flip(np.argsort(similarities), axis=0)
        recs, scores, top_tropes = [], [], []

        i = -1
        while len(recs) < 5:
            try:
                i += 1
                dot = np.ndarray.flatten(sparse_mbt[sorted_titles[i]].multiply(query_vec).A)
                top_tropes_of_result = top_tropes_from_vector(dot, n_tropes, col_to_trope_list)
                if len(top_tropes_of_result) < 5: continue
                recs.append(movie_titles[sorted_titles[i]])
                scores.append(similarities[sorted_titles[i]])
                top_tropes.append(top_tropes_of_result)
            except IndexError:
                break

    return recs, scores, top_tropes
