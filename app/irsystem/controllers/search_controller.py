from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.controllers.TVTropesScraper.TFIDF import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

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

    popularity = request.args.get('popSlide')

    q = None
    if(request.args.get('query')):
        q = request.args.get('query')

    if(not request.args.get('spec')):
        spec = "False"
    else:
        spec = dict()

    outputMessage = None
    output = []

    inspiration = None

    # initialize internal vars
    with open('app/irsystem/controllers/DatasetInfo/book_dataset.json') as json_file:
        booksJSON = json.load(json_file)
    with open('app/irsystem/controllers/DatasetInfo/movie_dataset.json') as json_file:
        moviesJSON = json.load(json_file)

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
    if q:
        outputMessage = "Your Search: " + q
        retrieval = find_relevant(datasets = datasets,
                                  inverted_indices = inverted_indices,
                                  query = q,
                                  input_category = queryType,
                                  result_category = outputType,
                                  min_df = 3,
                                  normalize = True,
                                  idf = "log",
                                  popularity_weight = popularity)

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
        inspiration = randomNInsp(moviesJSON, 10)
    if(queryType == "book"):
        inspiration = randomNInsp(booksJSON, 10)

    if spec == "False":
        if retrieval:
            i = 0
            for entry in retrieval[0]:
                output.append(dict())
                if queryType == "movie":
                    output[i]["title"] = entry[0]
                    if "author" in booksJSON.get(entry[0], ""):
                        output[i]["author"] = booksJSON[entry[0]]["author"]
                    output[i]["simScore"] = round(entry[1], 3)
                    output[i]["tropes"] = "".join(elem for elem in topNTropes(retrieval[1][entry[0]], 5))
                    if "reviews" in booksJSON.get(entry[0], ""):
                        output[i]["reviews"] = list()
                        for review in booksJSON[entry[0]]["reviews"]:
                            output[i]["reviews"].append(review)
                    if "img" in booksJSON.get(entry[0], ""):
                        output[i]["img"] = booksJSON[entry[0]]["img"]

                if queryType == "book":
                    output[i]["title"] = entry[0]
                    output[i]["simScore"] = round(entry[1], 3)
                    output[i]["tropes"] = "".join(elem for elem in topNTropes(retrieval[1][entry[0]], 5))
                    if entry[0] in moviesJSON and "img" in moviesJSON[entry[0]]:
                        output[i]["img"] = moviesJSON[entry[0]]["img"]

                i += 1
        elif(q):
            outputMessage = "Sorry, \'" + q + "\' is an invalid query."
    else:
        if(queryType == "movie"):
            spec["title"] = q
            if "author" in booksJSON[q]:
                spec["author"] = booksJSON[q]["author"]
            if "rating" in booksJSON[q]:
                spec["rating"] = booksJSON[q]["rating"]
            if "published" in booksJSON[q]:
                spec["year"] = booksJSON[q]["published"]
            if "summary" in booksJSON[q]:
                spec["summary"] = booksJSON[q]["summary"]
            if "reviews" in booksJSON[q]:
                spec["reviews"] = reversed(booksJSON[q]["reviews"])
            if "img" in booksJSON.get(q, ""):
                spec["img"] = booksJSON[q]["img"]

            spec["tropes"] = allTropes(book_tropes_data[q])
        else:
            spec["title"] = q.replace('\'', '%27')
            if q in moviesJSON:
                if "rating" in moviesJSON[q]:
                    spec["rating"] = moviesJSON[q]["rating"]
                if "published" in moviesJSON[q]:
                    spec["year"] = moviesJSON[q]["published"]
                if "summary" in moviesJSON[q]:
                    spec["summary"] = moviesJSON[q]["summary"]
                if "reviews" in moviesJSON[q]:
                    spec["reviews"] = reversed(moviesJSON[q]["reviews"])
                if "img" in moviesJSON[q]:
                    spec["img"] = moviesJSON[q]["img"]
            spec["tropes"] = allTropes(movie_tropes_data[q])

    # export
    return render_template('search.html', isHomeScreen = isHomeScreen, inspiration = inspiration, validQueries = validQueries, queryType = queryType, query = q, outputMessage = outputMessage, outputType = outputType , output = output, spec = spec)