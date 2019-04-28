from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import math
import pickle
import re
from collections import defaultdict
import gensim
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import itertools
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity as cosinesim
from tqdm import tqdm

def cosine_similarity(e,v):
    """
    #Input:
    #e = nxd input matrix with n row-vectors of dimensionality d (n is number of dictionary_keys)
    #v = mxd input matrix with m row-vectors of dimensionality d (m is number of test samples)
    # Output:
    # Matrix D of size nxm
    # s(i,j) is the cosinesimiarlity of embed(i,:) and test(j,:)
    """
    g=e.dot(v.T)
    b=np.expand_dims(np.linalg.norm(e,axis=1),1)+1e-16  # plus this small value to avoid division zero.
    a=np.expand_dims(np.linalg.norm(v,axis=1),1)+1e-16  # plus this small value to avoid division zero.
    s=np.divide(g,np.multiply(b,a.T))
    # ... until here
    return s.T
def findknn(D,k):
    """
   # D=cos_distance matrix
   # k = number of nearest neighbors to be found
   # flag =0 , recommend book
   # flag =1 , recommend movie
    
   # Output:
   # indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
   # dists = Euclidean distances to the respective nearest neighbors
    """
    
    m = D.shape[0]
    ind = np.argsort(D, axis=1)
    
    indices = ind[:,::-1][:,:k]
   # print(indices)
    r = np.array([_ for _ in range(m)], dtype=np.int)
    r = np.array([r] * k).T   
    dists = D[r,indices] 
    return indices,dists

def popularity_multiplier(z, strength=1): 
    """A multiplier between 1 to ~1.6 based on a z-score."""
    z += 4.5
    z = min(z, 7)
    z = max(z, 2)
    return strength*math.log(z/2.0)+1

def load_from_json(file_name):
    with open(file_name, "r") as fp:
            json_file=json.load(fp)
    return json_file


def flattened_list(list_of_lists):
    if list_of_lists is None:
        return None
    flattened = []
    for sublist in list_of_lists:
        for val in sublist:
            flattened.append(val)
    return flattened
def top_tropes_from_vector(v, n_tropes,col_to_trope_list):
    top_dot = np.argsort(-v)[0]

    top_tropes = []
    for i in top_dot[:n_tropes]:
        if v[0][i] != 0:
            top_tropes.append(col_to_trope_list[i])
    return top_tropes
def get_boosted_index_from_summary(query,direction,threshold=0.15):
    """
    # Input:
    # query : name of book or movie
    # k : number of recomendation 
    # threshold: boosting if summary tf-idf theshold exceeds the threshold default:0.2
    # direction: 
    # direction = 'mb' : movie - >  books
    # direction = 'bm' : book  - >  movies
    
    # Output:
    # index of documents to be boosted
    """
        
    if direction == "mb":
        input_data = movie_summary
        input_id2name = movie_id_to_name
        input_name2id = movie_name_to_id
        output_data =book_summary
        output_id2name = book_id_to_name
        ouput_name2id = book_name_to_id
    elif direction == "bm":
        input_data = book_summary
        input_id2name = book_id_to_name
        input_name2id = book_name_to_id
        output_data = movie_summary
        output_id2name = movie_id_to_name
        ouput_name2id = movie_name_to_id
    else:
        raise Exception("Input direction not defined !")
        
    query_vec = input_data[[input_name2id[query]]]
    

    sim = cosine_similarity(output_data,query_vec)
    
    
    boosted_indices= np.where(sim>=threshold)[1]

    return boosted_indices

'''
Returns the k tropes closest to the keyword 

Inputs: 
    - keyword: the keyword to search
    - dictionary: dictionary of trope words in dataset to search against 
    - word_to_trope: dictionary mapping word to set of tropes containing the word
    - model: the pretrained gensim model
    - top_k: number of tropes to return

Returns: 
    - trope_matches: list of the k tropes that are closest to the keyword
'''
def get_closest_tropes_to_keyword(keyword, dictionary, word_to_trope, model, top_k = 5): 
    
    # check that keyword is in vocabulary 
    if keyword in model.vocab: 
        # compute cosine similarity between query and all trope words
        all_words = list(dictionary.values())
        all_words = [word for word in all_words if word in model.vocab]
        dists = model.distances(keyword, all_words)

        # sort by similarity in ascending order (0 = perfect similarity)
        sorted_indices = np.argsort(dists)
        sorted_keyword_match = [all_words[idx] for idx in sorted_indices[:top_k]]

   #     print('\ntop {} matches most similar to `{}`'.format(top_k, keyword))
   #     for word in sorted_keyword_match: 
   #         print('`{}` : {}'.format(word, word_to_trope[word]))

        trope_matches = list(itertools.chain.from_iterable([word_to_trope[word] for word in sorted_keyword_match if word in word_to_trope]))
    #    print('\nenhancing search with : {}'.format(trope_matches[:top_k]))

        return trope_matches[:top_k]
    
    else: 
        print('`{}` not in model vocabulary, cannot enhance search with keyword'.format(keyword))
        return []
    
'''
Build a vectorizer and tf-idf matrix corresponding to a dataset

Inputs: 
    - data: dictionary mapping titles to tropes
    
Returns: 
    - vectorizer: a vectorizer object 
    - tf_idf_matrix: a tf-idf matrix corresponding to the topes in the dataset
'''
def make_tf_idf(data): 
    
    # make a vectorizer based on the 'to' dataset
    vectorizer = TfidfVectorizer(analyzer = 'word',
                                 tokenizer = lambda x : x, 
                                 lowercase = False)
    tf_idf_matrix = vectorizer.fit_transform(list(data.values()))

    return vectorizer, tf_idf_matrix
'''
build a dataset using tropes 

Inputs: 
    - data: dictionary mapping titles to list of tropes

Returns: 
    - dictionary: a bag of words dictionary representation tropes 
    - word_to_trope: mapping from individual word to tropes that contain the word
'''
def build_representation_for_tropes(data): 
    
    # build corpus of titles that contain field 
    corpus = []
    word_to_trope = {}
    for title, tropes in data.items(): 
        all_trope_words_for_title = []
        for trope in tropes: 
            trope_words = [word.lower() for word in re.findall('[A-Z][^A-Z]*', trope)]
            all_trope_words_for_title.extend(trope_words)
            for word in trope_words: 
                if word in word_to_trope: 
                    word_to_trope[word].add(trope)
                else: 
                    word_to_trope[word] = set([trope])
        corpus.append(all_trope_words_for_title)                                              
            
   
        
    # build dictionary from corpus
    dictionary = Dictionary(corpus)

    return dictionary, word_to_trope
def print_results(title, keyword, similarity_scores, to_dataset, top_k_titles = 10): 
    
    # sort the scores in descending order
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    # get list of titles
    to_titles = list(to_dataset.keys())

    print('\ntop {} most similar titles to `{}` by trope to keyword `{}` '.format(top_k_titles, title, keyword))
    for idx in range(top_k_titles): 
        print((similarity_scores[ranked_indices[idx]], to_titles[ranked_indices[idx]]))
'''
Finds the best titles according to tropes based on an input title and keyword. 
The trope words of the title being searched and the trope words most similar to 
the keyword are used to find the best matches. Half the number of the tropes corresponding
to the title being queried are used to enhance the keyword aspect of the search (eg. if the 
queried title has 10 tropes associated with it, then the top 5 tropes associated with the 
keyword will be used to enhance the search. This value seems to result in a good balance)

Inputs: 
    - title: the title being queried
    - from_dataset: the dataset corresponding to the title (eg. book dataset if book title)
    - keyword: the keyword to search
    - dictionary: dictionary representation of trope words associated with each title of desired return type
    - word_to_trope: dictionary mapping word to set of tropes containing the word
    - to_tf_idf_matrix: tf-idf representation of tropes associated with each title of desired return type
    - similarity_matrix: similarity matrix according to tf-idf representation 
    - model: the pretrained gensim model
    
Reutrns: 
    - similarity_scores: numpy array of similarity scores in where the index in the array corresponds
                         to the index in the dataset of the media type being recommended
'''
def best_titles_by_tropes_enhanced(title, from_dataset, keyword, dictionary, word_to_trope, vectorizer, to_tf_idf_matrix, model): 
    
    # get tropes coresponding to title 
    title_tropes =  from_dataset[title]
  #  print('tropes for title `{}` : {}'.format(title, title_tropes))
    
    # get most similar tropes to keyword, use half the number of tropes in the title to enhance search 
    top_k_tropes = int(len(title_tropes)/2)
    most_similar_tropes = get_closest_tropes_to_keyword(keyword, dictionary, word_to_trope, model, top_k_tropes)
    
    # extend query to include tropes associated with keyword
    query_tropes = most_similar_tropes + title_tropes
    
   # print('\ntropes used for final query : {}'.format(query_tropes))
    
    # generate a query vector 
    query_vector = vectorizer.transform([query_tropes])
    
    # compute cosine similarity between query and all titles 
    similarity_scores = cosinesim(query_vector, to_tf_idf_matrix).flatten()
    
    return similarity_scores
    
with open("./app/irsystem/controllers/TVTropesScraper/Film/Film_tropes_dataset3.json", 'r') as f:
    movie_tropes_data = json.load(f)
with open("./app/irsystem/controllers/TVTropesScraper/Literature/Literature_tropes_dataset3.json", 'r') as f:
    book_tropes_data = json.load(f)

with open("./app/irsystem/controllers/DatasetInfo/book_dataset.json", 'r', encoding='utf-8') as json_file:  
    alena_books = json.loads(json_file.read())
with open("./app/irsystem/controllers/DatasetInfo/movie_dataset.json", 'r', encoding='utf-8') as json_file:  
    alena_movies = json.loads(json_file.read())
movielens_reviews = pickle.load(open("./app/irsystem/controllers/DatasetInfo/movielens_reviews.p", "rb" ))


movie_id_to_summary=load_from_json("./app/irsystem/controllers/DatasetInfo/movie_summary.json")
book_id_to_summary=load_from_json("./app/irsystem/controllers/DatasetInfo/book_summary.json")
movie_summary_corpus= [" ".join(flattened_list(movie_id_to_summary[idx])) if movie_id_to_summary[idx] is not None else "" for idx in list(movie_id_to_summary.keys())]
book_summary_corpus= [" ".join(flattened_list(book_id_to_summary[idx]))  if book_id_to_summary[idx] is not None else "" for idx in list(book_id_to_summary.keys()) ]
# vecterize movie and book
vectorizer = TfidfVectorizer(sublinear_tf =True,smooth_idf=True,stop_words=None)
vectorizer.fit(movie_summary_corpus+book_summary_corpus)
movie_summary=vectorizer.transform(movie_summary_corpus).toarray()
book_summary=vectorizer.transform(book_summary_corpus).toarray()


inverted_index_books = defaultdict(list)
for book, trope_list in book_tropes_data.items():
    for trope in trope_list:
        inverted_index_books[trope].append(book)

inverted_index_movies = defaultdict(list)
for movie, trope_list in movie_tropes_data.items():
    for trope in trope_list:
        inverted_index_movies[trope].append(movie)

movie_titles = []
for k, v in alena_movies.items():
    movie_titles.append((k, v['idx']))
movie_titles.sort(key=lambda pair : pair[1])
movie_titles = [k[0] for k in movie_titles]

book_titles = []
for k, v in alena_books.items():
    book_titles.append((k, v['idx']))
book_titles.sort(key=lambda pair : pair[1])
book_titles = [k[0] for k in book_titles]


common_tropes = set(inverted_index_movies.keys()) | set(inverted_index_books.keys())
# common_tropes = {s.lower() for s in common_tropes}
tf_idf = TfidfVectorizer(min_df=3, lowercase=False, vocabulary = common_tropes, norm='l2', use_idf=True, binary=True)
movie_by_trope = tf_idf.fit_transform([' '.join(movie_tropes_data[movie_titles[i]]) for i in range(len(movie_titles))]).toarray()
book_by_trope = tf_idf.fit_transform([' '.join(book_tropes_data[book_titles[i]]) for i in range(len(book_titles))]).toarray()

trope_to_col = tf_idf.vocabulary_
col_to_trope_list = tf_idf.get_feature_names()

movie_name_to_id= {movie_titles[i]:i  for i in range(len(movie_titles))}
movie_id_to_name= {i:movie_titles[i]  for i in range(len(movie_titles))}
book_name_to_id= {book_titles[i]:i  for i in range(len(book_titles))}
book_id_to_name= {i:book_titles[i]  for i in range(len(book_titles))}



movies_popularity = np.zeros(len(movie_titles))
books_popularity = np.zeros(len(book_titles))

for j in range(len(movie_titles)):
    popularity_boost = 0
    if movie_titles[j] in movielens_reviews:
        z = (movielens_reviews[movie_titles[j]][0]-2000)/8000 # z-score of number of reviews
        popularity_boost += popularity_multiplier(z, strength=2)/5
        z = (movielens_reviews[movie_titles[j]][1]-3)/0.5  # z-score of 5-star rating
        popularity_boost += popularity_multiplier(z, strength=2)/5
    movies_popularity[j] = popularity_boost

for i in range(len(book_titles)):
    popularity_boost = 0
    if 'num_reviews' in alena_books[book_titles[i]]:
        z = (alena_books[book_titles[i]]['num_reviews']-54)/364
        popularity_boost += popularity_multiplier(z, strength=0.3)/2.2
    if 'rating' in alena_books[book_titles[i]]:
        z = (alena_books[book_titles[i]]['rating']-3)/0.5
        popularity_boost += popularity_multiplier(z, strength=0.3)/2.2
    books_popularity[i] = popularity_boost

model = KeyedVectors.load_word2vec_format('./app/irsystem/controllers/DatasetInfo/gensim_glove.6B.50d.txt', binary = False, limit=50000)
# create dictionary representations of datasets
book_dictionary, book_word_to_trope = build_representation_for_tropes(book_tropes_data)
movie_dictionary, movie_word_to_trope = build_representation_for_tropes(movie_tropes_data)
# build book to movie tf-idf matrix
book_to_movie_vectorizer, movie_tf_idf = make_tf_idf(movie_tropes_data)
# build movie to book tf-idf matrix
movie_to_book_vectorizer, book_tf_idf = make_tf_idf(book_tropes_data)

def is_not_blank(s):
    return bool(s and s.strip())

def recommendation(title, keyword=None,k=5,n_tropes=5,direction='mb', popularity_weight=0,boosting=True,relevance_feedback=False):
#     mod_mbt = np.where(movie_by_trope==0, -x, movie_by_trope*y)
#     mod_bbt = np.where(book_by_trope==0, -x*c, book_by_trope*y*c)
    """
    # Input:
    # query : name of book or movie
    # k : number of recomendation 
    # direction: 
    # direction = 'mb' : movie - >  books
    # direction = 'bm' : book  - >  movies
    # n_tropes: number of top tropes to be returned and displayed
    # popularity_weight: popularity weight
    # boosting : apply boosting to tf-idf tropes using tf-idf summary
    # relevance_feedback 
    # keyword: the keyword to search
    
    # Output:
    # recomendations: name of top k of recommended results
    # recomendations_scores : scores of top k of recommended results
     # recomendations_scores : a nested list of top tropes returned of size : (k * n_tropes)
    """
    if popularity_weight is None: popularity_weight = 0
    popularity_weight = float(popularity_weight)
    
    if direction=='mb':
        from_data = movie_tropes_data
        to_data=book_tropes_data
        input_data = movie_by_trope
        input_id2name = movie_id_to_name
        input_name2id = movie_name_to_id
        output_data = book_by_trope
        output_id2name = book_id_to_name
        ouput_name2id = book_name_to_id
        popularity=books_popularity
        word_to_trope = book_word_to_trope
        key_word_vectorizer = movie_to_book_vectorizer
        key_word_tf_idf_matrix = book_tf_idf
        
       
    elif direction == "bm":
        from_data = book_tropes_data
        to_data=movie_tropes_data
        input_data = book_by_trope
        input_id2name = book_id_to_name
        input_name2id = book_name_to_id
        output_data = movie_by_trope
        word_to_trope = movie_word_to_trope
        output_id2name = movie_id_to_name
        word_to_trope = book_word_to_trope
        ouput_name2id = movie_name_to_id
        popularity=movies_popularity
        dictionary = movie_dictionary
        word_to_trope = movie_word_to_trope
        key_word_vectorizer = book_to_movie_vectorizer
        key_word_tf_idf_matrix = movie_tf_idf
        
    else:
        raise Exception("Input direction not defined !")

        
    if keyword is not None and is_not_blank(keyword):
        
        key_word_scores = best_titles_by_tropes_enhanced(title, 
                                                       from_data, 
                                                       keyword, 
                                                       dictionary, 
                                                       word_to_trope, 
                                                       key_word_vectorizer, 
                                                       key_word_tf_idf_matrix, 
                                                       model)
        # example to print nice list of ranked results 
        #print_results(title, keyword, key_word_scores, to_data, 10)
   
    query_vec = input_data[[input_name2id[title]]]
    
   
    sim = cosine_similarity(output_data,query_vec)

    if popularity_weight > 0:
        sim = np.multiply(sim, popularity_weight * popularity)

    if relevance_feedback:
        
        indices,scores = findknn(sim,k)
        
        alpha = 1
        beta = 0.75
        gamma = 0.15
        top_k=2 # choose top 2 as relevant query
        
        def get_irrevalent(sim,threshold=0):
            """
            # Similarity score <= threshold will be consider as irrelevant docs
            
            """
            m = sim.shape[0]
            
            ind = np.argsort(sim, axis=1)
            
            ire_ind = np.where(sim<=0)[1]
            
            return ire_ind
        
        irrelevant_docs_ids = get_irrevalent(sim)

        relevant_docs_ids = indices[0][:top_k] 
   
        modified_query_vec =   alpha * query_vec  \
                             + beta * np.sum(output_data[relevant_docs_ids],axis=0,keepdims=True)/len(relevant_docs_ids) \
                             - gamma * np.sum(output_data[relevant_docs_ids],axis=0,keepdims=True)/len(irrelevant_docs_ids) 
        
        
        query_vec = modified_query_vec
        sim = cosine_similarity(output_data,query_vec)

        indices,scores = findknn(sim,k)
        
        
    if boosting :
        boosted_score=0.1
        boosted_idx=get_boosted_index_from_summary(title,direction=direction,threshold=0.2)
        
        if boosted_idx is not None:
            for idx in boosted_idx:
                 sim[0][idx]=min(sim[0][idx]+boosted_score,1.0) 
    

    no_key_word_score=sim[0]
 
#     print(no_key_word_score)
#     print(key_word_scores)
  
    if keyword is not None and is_not_blank(keyword):
        print(keyword)
        scores=0.9*key_word_scores+0.1*no_key_word_score
    else:
        scores=no_key_word_score
        
    
   
    ranked_indices,ranked_scores=findknn(np.reshape(scores,newshape=(1,-1)),k)
    
    
    recomendations=[]
    recomendation_scores=[]
    top_tropes=[]
    for i in range(len(ranked_indices[0])):
        print ("{} \x1b[31m{:.3f}\x1b[0m".format(output_id2name[ranked_indices[0][i]], ranked_scores[0][i])) 
        # print(["".join(elem for elem in topNTropes(retrieval[1].get(entry[0]), 5))])
        recomendations.append(output_id2name[ranked_indices[0][i]])
        recomendation_scores.append(ranked_scores[0][i])
        dot=np.multiply(movie_by_trope[[ranked_indices[0][i]]], query_vec[0])
        tropes = top_tropes_from_vector(dot,n_tropes,col_to_trope_list)
        top_tropes.append(tropes)
        
        print(tropes)
    
    return recomendations,recomendation_scores,top_tropes

# example query 
recomendations,recomendation_scores,top_tropes=recommendation("Harry Potter and the Chamber of Secrets", keyword='monster',k=10,n_tropes=5,direction='bm', popularity_weight=0,boosting=True,relevance_feedback=True)