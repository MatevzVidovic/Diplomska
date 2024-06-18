

from data_preparation import prepare_data
from naloga5 import WrapperModel


import gzip
import json

import pickle

# ohe_cutoff = 1000
# tfidf_cutoff = 1000


# ohe_cutoff = 150
# tfidf_cutoff = 150

# ohe_cutoff = 250
# tfidf_cutoff = 250

# ohe_cutoff = 30000
# tfidf_cutoff = 30000



hyper_parameters = {

    # "max_iter" : 400,

    # "URLs" : ohe_cutoff,
    # "authors" : ohe_cutoff,
    # "leads" : ohe_cutoff,
    # "keywords" : tfidf_cutoff,
    # "gpt_keywords" : tfidf_cutoff,
    # "topics" : tfidf_cutoff, 
}



# Path to your .json.gzip file
file_path = './data/rtvslo_train.json.gz'

# Open the gzip file
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    # Read and parse the JSON data
    data = json.load(f)


curr_model = WrapperModel(data, hyper_parameters)

with open('wrapper_model.pkl', 'wb') as f:
    pickle.dump(curr_model, f)