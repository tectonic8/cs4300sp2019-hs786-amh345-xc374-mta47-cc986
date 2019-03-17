from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "The Book Was Better"
net_id = "Hartek Sabharwal (hs786), "+"Alena Hutchinson (amh345), " +"Xingyu Chen (xc374), " +"Morgan Aloia (mta47)"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query
		data = range(5)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



