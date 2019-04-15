from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.controllers.TVTropesScraper.TFIDF import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

@irsystem.route('/', methods=['GET'])
def search():
	queryType = request.args.get('searchType')
	query = request.args.get('search')
	data = []
	output_message = ''

	if query:
		output_message = "Your search: " + query
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
		# data[i][0] = Title, data[i][1] = SimScore, data[i][2] = (Trope, RelScore)
		i = 0
		
		for entry in retrieval[0]:
			data.append([])
			data[i].append(entry[0])
			data[i].append(entry[1])
			data[i].append(retrieval[1][entry[0]])
			i += 1
		print(data)
		
	return render_template('search.html', output_message=output_message, data=data)



