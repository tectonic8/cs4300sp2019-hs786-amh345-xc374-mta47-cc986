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
		queryType = request.args.get('queryType')
	
	if(queryType == "movie"):
		outputType = "book"
	else:
		outputType = "movie"

	query = None
	if(request.args.get('query')):
		query = request.args.get('query')

	outputMessage = None
	output = []

	inspiration = None

	# initialize internal vars
	with open('app/irsystem/controllers/DatasetInfo/tbwb_book_dataset.json') as json_file:
	    booksJSON = json.load(json_file)
	with open('app/irsystem/controllers/DatasetInfo/tbwb_movie_dataset.json') as json_file:
	    moviesJSON = json.load(json_file)

	for book in books:
		if(book.lower() in booksJSON):
			booksJSON[book] = booksJSON.pop(book.lower())
	for movie in movies:
		if(movie.lower() in moviesJSON):
			moviesJSON[movie] = moviesJSON.pop(movie.lower())

	retrieval = None

	# run query
	if(query):
		outputMessage = "Your Search: " + query
		retrieval = find_relevant(datasets = datasets,
                                  inverted_indices = inverted_indices,
                                  query = query, 
                                  input_category = queryType,
                                  result_category = outputType,
                                  min_df = 3,
                                  normalize = True,
                                  idf = "log")

	# set export vars
	if(not request.args.get('query')):
		isHomeScreen = True
	else:
		isHomeScreen = False

	if(queryType == "movie"):
		inspiration = randomNInsp(moviesJSON, 10)
	if(queryType == "book"):
		inspiration = randomNInsp(booksJSON, 10)

	if retrieval:
		i = 0
		for entry in retrieval[0]:
			output.append(dict())
			if(queryType == "movie"):
				output[i]["title"] = entry[0]
				if "author" in booksJSON[entry[0]]:
					output[i]["author"] = booksJSON[entry[0]]["author"]
				output[i]["simScore"] = entry[1]
				output[i]["tropes"] = "".join(elem for elem in topNTropes(retrieval[1][entry[0]], 5))
			if(queryType == "book"):
				output[i]["title"] = entry[0]
				output[i]["simScore"] = entry[1]
				output[i]["tropes"] = "".join(elem for elem in topNTropes(retrieval[1][entry[0]], 5))
			i += 1
	elif(query):
		outputMessage = "Sorry, \'" + query + "\' is an invalid query."

	# export
	return render_template('search.html', isHomeScreen = isHomeScreen, inspiration = inspiration, queryType = queryType, query = query, outputMessage = outputMessage, outputType = outputType , output = output)