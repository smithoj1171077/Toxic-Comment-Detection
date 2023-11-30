# importing libraries

import torch
import pandas as pd
import numpy as np
import keras
from torch import nn as nn
import keras.preprocessing.text as text
import keras.preprocessing.sequence as sequence
from data_import import X_train, X_test, embeddings_index
from torch.nn import Embedding
from torch.nn import Dropout1d
from torch.nn import LSTM
from torch.nn import Conv1d
import torch.utils

# define the maximum number of tokens to have an embedding vector computed
max_features = 100000
maxlen = 150
# make each token represented by a vector with 300 dimensions
embed_size = 300

# replace the Nans comments with an empty space
X_train.fillna(' ')
X_test.fillna(' ')

# convert all comment text to lower casing
X_train = X_train['comment_train'].str.lower()
X_test = X_test['comment_test'].str.lower()

# Text vectorization and GloVe embeddings
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
# create a mapping of words in the vocab generated from the datasets to an arbitrary integer index value
tokenizer.fit_on_texts(list(X_train))

"""transform the texts in the datasets to a set of integer sequences 
where each word has been mapped to it's index value"""
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# ensure that all sequences as of same length by padding with zeros
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# the embedding matrix is
word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue

    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# convert to tensor so it can be input into model
embedding_tensor = torch.from_numpy(embedding_matrix)


class BiLSTMCNN(nn.Module):
    """This class is the pytorch implementation of the Bi-LSTM-CNN model"""
    def __init__(self, max_features, embed_size, embedding_tensor):
        super().__init__()
        # Define an object to input our embeddings into the pytorch model
        self.embedding = Embedding(embedding_tensor,max_features,embed_size)
        """To prevent over fitting, we need to drop out some of the filters in training,
        each filter (if trained correctly) should represent a particular pattern in the word embeddings 
        known as a channel generally the proportion dropped out should be somewhere between 
        20%-50%, 35% is a good middle ground between over fitted and under fitted. 
        """
        self.dropout = Dropout1d(p=0.35)
        """ LSTM with bidirectional layering"""

        """Our LSTM has 128 units, each unit has a cell which contains a value called a cell state, 
        the cell state can only be modified by percentage factor determined by the forget gate and/or
        it's sum with a value determined by the input gate. The model weights can not directly impact the 
        cell state (which represents long term trends) thus the model is not affected by the vanishing/exploding
        gradient problem which often limits more basic RNNs. Since language involves complex sequential patterns, 
        using 128 unit's is appropriate since each unit's weights can be tuned to recognise a different 
        temporal/sequential dependency."""
        # pytorch does not support recurrent dropout
        self.bi_lstm = LSTM(input_size=128, dropout=0.15, bidirectional=True)
        self.conv1d = Conv1d(64, kernel_size=3, padding='valid', out_channels=64)
        # max pooling is great at capturing local features, while average pooling is good for global trend capturing
        # concatenate layers to get both benefits
        self.avg_pool = nn.AvgPool1d(kernel_size=3)
        self.max_pool = nn.MaxPool1d(kernel_size=3)
        self.output_layer = nn.Linear(200,6)
    # training of the network is defined here

    def forward(self, input):
        input_lt = torch.LongTensor(input)
        input_embedded = self.embedding(input_lt)
        input_after_drop_out = self.dropout(input_embedded)
        bi_lstm_output = self.bi_lstm(input_after_drop_out)
        cnn_out = self.conv1d(bi_lstm_output)
        avg_pool_results = self.avg_pool(cnn_out)
        max_pool_results = self.max_pool(cnn_out)
        concatenated_results = torch.cat((avg_pool_results, max_pool_results))
        output = self.output_layer(concatenated_results)
        prediction = nn.functional.softmax(output)
        return prediction

# prep the data for training
trainloader = torch.utils.data.DataLoader(embedding_tensor, batch_size=32, shuffle=True, num_worders=2)



model = BiLSTMCNN(max_features, embed_size, embedding_tensor)

# define cost function as binary cross entropy loss
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        inputs,labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished training')




















