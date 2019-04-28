# Methods to compose HTTP response JSON 
from flask import jsonify
import base64
import json
import operator
import re
from random import shuffle
import numpy as np 

with open('app/irsystem/controllers/TVTropesScraper/Main/tropes_description_dataset.json') as json_file:
        tropeDescriptions = json.load(json_file)

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

def allTropes(l):
    out = []
    i = 0
    for trope in l:
        trope = re.sub(r"([A-Z])([A-Z])", r"\1 \2", trope)
        trope = re.sub(r"(\w)([A-Z])", r"\1 \2", trope)
        if (i == len(l)-1):
            out.append((trope, tropeDescriptions[trope.replace(" ", "")]))
        else:
            out.append((trope + ", ", tropeDescriptions[trope.replace(" ", "")]))
        i += 1

    return out

def topNTropes(d, n):
    top = []

    i = 0
    while(i < n):
        v=list(d.values())
        k=list(d.keys())
        m = k[v.index(max(v))]
        d.pop(m)

        if(i == n-1):
            m = re.sub(r"(\w)([A-Z])", r"\1 \2", m)
            m = re.sub(r"([A-Z])([A-Z])", r"\1 \2", m)
            top.append((m, tropeDescriptions[m.replace(" ", "")]))

        else:
            m = re.sub(r"(\w)([A-Z])", r"\1 \2", m) + ", "
            m = re.sub(r"([A-Z])([A-Z])", r"\1 \2", m)
            top.append((m, tropeDescriptions[m[:-2].replace(" ", "")]))

        i += 1

    return top

def randomNInsp(d, n):
    valid = [(item[0], item[0].replace("'", "%27"))
             for item in d.items()
             if item[1].get("rating", 0) >= 4.5]
    shuffle(valid)
    return valid[:n]


def auto_paragraph(text):
    """
    Autoparagraphs the given text by separating the text into a series of paragraphs.
    Returns a list of paragraphs.
    :param text:
    :return:
    """

    paras = []

    lines = text.strip().split(". ")
    para = []

    while len(lines) > 0:
        line = lines.pop(0)
        if len(line.strip()) == 0: continue
        para.append(line)
        if len(para) > 1 and len(para) % 4 == 0:
            para_text = ". ".join(para)
            if not para_text.endswith("."): para_text += "."
            paras.append(para_text)
            para = []

    if len(para) > 0:
        final_para = ". ".join(para)
        if not final_para.endswith("."): final_para += "."
        paras.append(final_para)

    return paras

