# Methods to compose HTTP response JSON 
from flask import jsonify
import base64
import json
import operator
import re
from random import shuffle
import numpy as np 

def http_json(result, bool):
	result.update({ "success": bool })
	return jsonify(result)


def http_resource(result, name, bool=True):
	resp = { "data": { name : result }}
	return http_json(resp, bool)


def http_errors(result): 
	errors = { "data" : { "errors" : result.errors["_schema"] }}
	return http_json(errors, False)

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict 
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)
        
def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def topNTropes(d, n):
    top = []

    i = 0
    print(d.get)
    while(i < n):
        v=list(d.values())
        k=list(d.keys())
        m = k[v.index(max(v))]
        d.pop(m)

        if(i == n-1):
            m = re.sub(r"(\w)([A-Z])", r"\1 \2", m)
            m = re.sub(r"([A-Z])([A-Z])", r"\1 \2", m)

        else:
            m = re.sub(r"(\w)([A-Z])", r"\1 \2", m) + ", "
            m = re.sub(r"([A-Z])([A-Z])", r"\1 \2", m)
        top.append(m)

        i += 1

    return top

def randomNInsp(d, n):
    a = []
    for item in d.items():
        if("rating" in item[1] and item[1]["rating"] >= 4.5):
            a.append(item[0])
    shuffle(a)
    while (len(a) > n):
        a.pop()
    return a
