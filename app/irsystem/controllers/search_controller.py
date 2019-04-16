from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.controllers.TVTropesScraper.TFIDF import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

@irsystem.route('/', methods=['GET'])
def search():
	with open('app/irsystem/controllers/DatasetInfo/tbwb_book_dataset.json') as json_file:
	    booksJSON = json.load(json_file)
	with open('app/irsystem/controllers/DatasetInfo/tbwb_movie_dataset.json') as json_file:
	    moviesJSON = json.load(json_file)

	queryType = request.args.get('searchType')
	query = request.args.get('search')
	data = []
	output_message = ''
	invQ = "FirstLoad"
	r = None
	if(queryType == "MB"  or r is None):
		r = randomNInsp(moviesJSON, 10)
	if(queryType == "BM"):
		r = randomNInsp(booksJSON, 10)

	if query:
		output_message = "Your Search: " + query
		if(queryType == "MB"):
			retrieval = find_relevant(datasets=datasets,
                                      inverted_indices=inverted_indices,
                                      query=query, 
                                      input_category="movie",
                                      result_category="book",
                                      min_df=3,
                                      normalize=True,
                                      idf="log"
                                     )

		if(queryType == "BM"):
			retrieval = find_relevant(datasets=datasets,
	                                      inverted_indices=inverted_indices,
	                                      query=query, 
	                                      input_category="book",
	                                      result_category="movie",
	                                      min_df=3,
	                                      normalize=True,
	                                      idf="log"
	                                     )
			
			
		# data[i][0] = Title, data[i][1] = Author, data[i][2] = SimScore, data[i][3] = (Trope, RelScore)
		if retrieval:
			invQ = False
			i = 0
			for entry in retrieval[0]:
				data.append([])
				print(entry[0])
				data[i].append(entry[0])
				if queryType == "MB" and "author" in booksJSON[entry[0].lower()]:
					data[i].append(booksJSON[entry[0].lower()]["author"])
				else:
					data[i].append("Author Not Listed")
				data[i].append(entry[1])
				data[i].append(topNTropes(retrieval[1][entry[0]], 5))

				i += 1
		elif(queryType):
			invQ = True
			output_message = "Sorry, \'" + query + "\' is an invalid query."
	return render_template('search.html', insp = r, output_message = output_message, data = data, invalidQuery = invQ)