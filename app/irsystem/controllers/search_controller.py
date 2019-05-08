from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.controllers.combined.recommender import *

# initialize internal vars
with open('app/irsystem/controllers/DatasetInfo/book_dataset.json') as json_file:
    booksJSON = json.load(json_file)
with open('app/irsystem/controllers/DatasetInfo/movie_dataset.json') as json_file:
    moviesJSON = json.load(json_file)

@irsystem.route('/', methods=['GET'])
def search():
    # initialize export vars
    if(not request.args.get('queryType')):
        queryType = "movie"
    else:
        if "/" in request.args.get('queryType'):
            queryType = request.args.get('queryType')[:-1]
        else:
            queryType = request.args.get('queryType')

    if(queryType == "movie"):
        outputType = "book"
    else:
        outputType = "movie"

    q = None
    if(request.args.get('query')):
        q = request.args.get('query')
        
    rel_titles = []
    for i in range(1, 6):
        title = request.args.get('rocchio' + str(i), None)
        if title is not None:
            rel_titles.append(title)
        
    k = None
    if(request.args.get('keyword')):
        k = request.args.get('keyword')

    popularity = request.args.get('popSlide')

    if(not request.args.get('spec')):
        spec = "False"
    else:
        spec = dict()

    pastSearch = None
    failedSearch = None
    pastKeyword = None
    pastPop = None
    output = []

    inspiration = None
    
    # books_lower_to_proper = {title.lower(): title for title in books}
    # movies_lower_to_proper = {title.lower(): title for title in movies}
    #
    # for book in booksJSON.keys():
    #     booksJSON[books_lower_to_proper.get(book.lower(), book)] = booksJSON.pop(book)
    #
    # for movie in moviesJSON.keys():
    #     moviesJSON[movies_lower_to_proper.get(movie.lower(), movie)] = moviesJSON.pop(movie)

    retrieval = None

    # run query
    if q and spec == "False":
        pastSearch = q
        if k:
            pastKeyword = k
        if popularity:
            pastPop = popularity

        direction = 'bm' if queryType == "book" else 'mb'


        results = find_relevant(q, keyword=k, n_recs=5, n_tropes=5,
                                                  direction=direction,
                                                  popularity_weight=popularity,
                                                  rocchio_titles = rel_titles)

        if results:
            recs, scores, top_tropes = results
            title_to_tropes = {recs[i]: top_tropes[i] for i in range(len(recs))}
            retrieval = ([(recs[i], scores[i]) for i in range(len(recs))], title_to_tropes)
        else:
            retrieval = False


    # set export vars
    validQueries = ""
    if(queryType == "movie"):
        validQueries = list(movies)
    else:
        validQueries = list(books)

    if(not request.args.get('query')):
        isHomeScreen = True
    else:
        isHomeScreen = False

    if(queryType == "movie"):
        inspiration = randomNInsp(moviesJSON, 3)
    if(queryType == "book"):
        inspiration = randomNInsp(booksJSON, 3)

    if spec == "False":
        if retrieval:
            i = 0
            for title, score in retrieval[0]:
                output.append(dict())
                if queryType == "movie":
                    output[i]["title"] = title
                    if "author" in booksJSON.get(title, ""):
                        output[i]["author"] = booksJSON[title]["author"]
                    output[i]["simScore"] = round(score, 3)
                    output[i]["tropes"] = topNTropes(retrieval[1][title], 5, title)
                    if "img" in booksJSON.get(title, ""):
                        output[i]["img"] = booksJSON[title]["img"]

                if queryType == "book":
                    output[i]["title"] = title
                    output[i]["simScore"] = round(score, 3)
                    output[i]["tropes"] = topNTropes(retrieval[1][title], 5, title)
                    if title in moviesJSON and "img" in moviesJSON[title]:
                        output[i]["img"] = moviesJSON[title]["img"]

                i += 1
            if len(retrieval[0]) == 0:
                failedSearch = "Sorry, insufficient data on '{}' for recommendations to be made.".format(q)
        elif (q):
            failedSearch = "Sorry, \'" + q + "\' is an invalid query."
    else:
        if(queryType == "movie"):
            spec["title"] = q
            if "author" in booksJSON[q]:
                spec["author"] = booksJSON[q]["author"]
            if "rating" in booksJSON[q]:
                spec["rating"] = str(round(float(booksJSON[q]["rating"])*4)/4)
            if "published" in booksJSON[q]:
                spec["year"] = booksJSON[q]["published"]
            if "summary" in booksJSON[q]:
                spec["summary"] = auto_paragraph(booksJSON[q]["summary"])
            if "reviews" in booksJSON[q]:
                spec["reviews"] = booksJSON[q]["reviews"][::-1]
            if "img" in booksJSON.get(q, ""):
                spec["img"] = booksJSON[q]["img"]

            spec["tropes"] = trope_with_descriptions(book_tropes_data[q])
        else:
            spec["title"] = q
            if q in moviesJSON:
                if "rating" in moviesJSON[q]:
                    spec["rating"] = str(round(float(moviesJSON[q]["rating"])*4)/4)
                if "published" in moviesJSON[q]:
                    spec["year"] = moviesJSON[q]["published"]
                if "summary" in moviesJSON[q]:
                    spec["summary"] = auto_paragraph(moviesJSON[q]["summary"])
                if "reviews" in moviesJSON[q]:
                    spec["reviews"] = moviesJSON[q]["reviews"][::-1]
                if "img" in moviesJSON[q]:
                    spec["img"] = moviesJSON[q]["img"]
            spec["tropes"] = trope_with_descriptions(movie_tropes_data[q])

    # export
    return render_template('search.html', 
                            isHomeScreen = isHomeScreen, 
                            inspiration = inspiration, 
                            validQueries = validQueries, 
                            queryType = queryType, 
                            query = q, 
                            pastSearch = pastSearch,
                            failedSearch = failedSearch,
                            pastKeyword = pastKeyword,
                            pastPop = pastPop,
                            outputType = outputType , 
                            output = output, 
                            spec = spec)