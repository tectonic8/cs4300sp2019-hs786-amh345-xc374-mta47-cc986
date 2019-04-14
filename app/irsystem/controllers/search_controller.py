from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.controllers.TVTropesScraper.TFIDF import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query
		data = find_relevant(datasets=datasets,
                                      inverted_indices=inverted_indices,
                                      query=query, 
                                      input_category="movie",
                                      result_category="book",
                                      min_df=3,
                                      normalize=True,
                                      idf="log"
                                     )
	return render_template('search.html', output_message=output_message, data=data)



