"""This module imports data, so we don't have to run that every time"""
import joblib
import numpy as np
import pandas as pd

# import the basic train and test data
X_train = pd.read_csv('train_preprocessed.csv')
X_test = pd.read_csv('test_preprocessed.csv')

# get the file path for the GloVe vectors
EMBEDDINGS_PATH = 'glove.840B.300d.txt'
# which is an unsupervised embedding method which creates token vectors such that their dot products are
# equal to the log probability of co-occurrence.

try:
    embeddings_index = joblib.load('embeddings_index.joblib')
except FileNotFoundError:

    # the format of the file is word_0, embedding_vector_values, word_1, embedding_vector_values, ...
    embeddings_index = {}
    print("starting import")
    with open(EMBEDDINGS_PATH, encoding='utf8') as f:
        for i, line in enumerate(f):
            if i > 5000:
                break
            print("line ", i)
            values = line.rstrip().rsplit(' ')
            word = values[0]
            # Convert embedding values to 32-bit floats since pytorch has a default float size of 32 bits
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    joblib.dump(embeddings_index, 'embeddings_index.joblib')
    embeddings_index = joblib.load('embeddings_index.joblib')
    print("data import complete")
print(embeddings_index)



