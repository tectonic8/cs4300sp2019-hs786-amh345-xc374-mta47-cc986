import tarfile
import ast
import gzip
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import math
import re
import csv
import editdistance
from collections import defaultdict
import os
from typing import Dict, List, Tuple


with open("app/irsystem/controllers/TVTropesScraper/Film/Film_tropes_dataset3.json", 'r') as f:
    movie_tropes_data = json.load(f)
with open("app/irsystem/controllers/TVTropesScraper/Literature/Literature_tropes_dataset3.json", 'r') as f:
    book_tropes_data = json.load(f)


books = list(book_tropes_data.keys())
movies = list(movie_tropes_data.keys())

inverted_index_books = defaultdict(list)
for book, trope_list in book_tropes_data.items():
    for trope in trope_list:
        inverted_index_books[trope].append(book)

inverted_index_movies = defaultdict(list)
for movie, trope_list in movie_tropes_data.items():
    for trope in trope_list:
        inverted_index_movies[trope].append(movie)

datasets = [movie_tropes_data, book_tropes_data]
inverted_indices = [inverted_index_movies, inverted_index_books]


def doc_norm(tropes_data,
             inverted_index,
             idf: str=None):
    """
    Note the custom formulae for normalization: avoids rewarding when norms[document] is small (e.g. <1)
    """
    if idf == "inverse":
        f = lambda trope: (1.0 / len(inverted_index[trope])) **2
    elif idf == "log":
        f = lambda trope: (1.0/(1+np.log(len(inverted_index[trope]))))**2
    elif idf is None:
        f = lambda trope: 1
    else:
        raise Exception("Invalid IDF")

    norms = defaultdict(int)
    for document, trope_list in tropes_data.items():
        for trope in trope_list:
            norms[document] += f(trope)
        norms[document] = math.sqrt(norms[document])
    return norms

def get_idf_func(input_inverted_index, result_inverted_index, idf: str):
    if idf == "inverse":
        return lambda trope: (1.0 / len(input_inverted_index[trope])) * (1.0 / len(result_inverted_index[trope]))
    elif idf == "log":
        return lambda trope: (1.0/(1+np.log(len(input_inverted_index[trope])))) * (1.0/(1+np.log(len(result_inverted_index[trope]))))
    elif idf is None:
        return lambda trope: 1
    else:
        raise Exception("Invalid IDF")

def filter_with_num_tropes(doc_scores: List[Tuple],
                           trope_contributions: Dict[str, Dict[str, int]],
                           num_tropes: int):
    """
    Exclude documents where number of similar tropes is <= [num_tropes]
    """
    return list(filter(lambda ds: len(trope_contributions[ds[0]]) >= num_tropes, doc_scores))

def find_relevant(datasets: List[Dict],
                  inverted_indices: List[Dict],
                  query: str,
                  input_category: str,
                  result_category: str,
                  normalize: bool=True,
                  idf:str=None,
                  min_df:int=0
                 ):
    """
    THE main TF-IDF function

    """
    idx = {"movie": 0, "book": 1}

    input_idx = idx[input_category]
    result_idx = idx[result_category]

    input_dataset = datasets[input_idx]

    f = get_idf_func(input_inverted_index=inverted_indices[input_idx],
                     result_inverted_index=inverted_indices[result_idx],
                     idf=idf)

    # Correcting search query to database title
    if query not in input_dataset:
        print("Could not find title: {}".format(query))
        return

    query_vec = input_dataset[query]

    doc_scores = defaultdict(int)

    trope_contributions = defaultdict(dict)
    # record weightage of each trope contributions

    # Update accumulators
    for trope in query_vec:
        if len(inverted_indices[input_idx][trope]) < min_df or len(inverted_indices[result_idx][trope]) < min_df:
            continue

        postings = inverted_indices[result_idx][trope]
        for doc in postings:
            weight_update = f(trope)
            doc_scores[doc] += weight_update
            trope_contributions[doc][trope] = weight_update

    # Normalize
    if normalize:
        norms = doc_norm(datasets[result_idx],
                 inverted_indices[result_idx],
                 idf=idf)
        for d in doc_scores:
            if norms[d] != 0:
                doc_scores[d] /= norms[d]

    doc_idx_scores = sorted(doc_scores.items(), key=lambda x:x[1], reverse=True)
    doc_scores = [(doc, score) for doc, score in doc_idx_scores if score > 0]

    doc_scores = filter_with_num_tropes(doc_scores, trope_contributions, num_tropes=5)
    return doc_scores[:10], trope_contributions
