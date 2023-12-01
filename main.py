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
import os

# define the maximum number of tokens to have an embedding vector computed
max_features = 100000
maxlen = 150
# make each token represented by a vector with 300 dimensions
embed_size = 300

# replace the Nans comments with an empty space
X_train = X_train.fillna(' ')
X_test = X_test.fillna(' ')
y_train = X_train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
# convert all comment text to lower casing
X_train = X_train['comment_text'].str.lower()
X_test = X_test['comment_text'].str.lower()

# Text vectorization and GloVe embeddings
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
# create a mapping of words in the vocab generated from the datasets to an arbitrary integer index value
tokenizer.fit_on_texts(list(X_train))
print(X_test)
print(X_train)
"""transform the texts in the datasets to a set of integer sequences 
where each word has been mapped to it's index value"""
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# ensure that all sequences as of same length by padding with zeros
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
X_train = X_train.astype(np.int64)
X_test = X_test.astype(np.int64)
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
embedding_tensor = torch.from_numpy(embedding_matrix.astype(np.float32))


class BiLSTMCNN(nn.Module):
    """This class is the pytorch implementation of the Bi-LSTM-CNN model"""
    def __init__(self, max_features, embed_size, embedding_tensor):
        super().__init__()
        # Define an object to input our embeddings into the pytorch model
        self.embedding = Embedding(max_features,embedding_dim=embed_size,_weight=embedding_tensor)
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
        self.bi_lstm = LSTM(input_size=embed_size,hidden_size=128, dropout=0.15, bidirectional=True, batch_first=True)
        self.conv1d = Conv1d(in_channels=128*2, kernel_size=3, padding='valid', out_channels=64)
        # max pooling is great at capturing local features, while average pooling is good for global trend capturing
        # concatenate layers to get both benefits
        self.avg_pool = nn.AvgPool1d(kernel_size=3)
        self.max_pool = nn.MaxPool1d(kernel_size=3)
        self.output_layer = nn.Linear(6272, 6)
    # training of the network is defined here

    def forward(self, input):
        input_lt = torch.LongTensor(input)
        input_embedded = self.embedding(input_lt)
        input_after_drop_out = self.dropout(input_embedded)
        bi_lstm_output, _ = self.bi_lstm(input_after_drop_out)
        bi_lstm_output = torch.cat((bi_lstm_output[:, :, :128], bi_lstm_output[:, :, 128:]), dim=2)
        bi_lstm_output = bi_lstm_output.permute(0, 2, 1)
        cnn_out = self.conv1d(bi_lstm_output)
        avg_pool_results = self.avg_pool(cnn_out)
        max_pool_results = self.max_pool(cnn_out)
        concatenated_results = torch.cat((avg_pool_results, max_pool_results),dim=1)
        output = self.output_layer(concatenated_results.view(concatenated_results.size(0), -1))
        return nn.functional.softmax(output)

def train_BiLSTMCNN():
    # prep the data for training
    trainloader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True, num_workers=2)

    model = BiLSTMCNN(max_features, embed_size, embedding_tensor)

    # define cost function as binary cross entropy loss
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            print(outputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished training')
    return model.state_dict

if __name__ == '__main__':
    if os.path.exists('model.pth'):
        model = BiLSTMCNN(max_features, embed_size, embedding_tensor)
        model.load_state_dict(torch.load('model.pth'))
    else:
         # create and save state dict
         model = BiLSTMCNN(max_features, embed_size, embedding_tensor)
         output = train_BiLSTMCNN()
         model.state_dict = output
         torch.save(output, 'model.pth')
    print(model.eval())



















