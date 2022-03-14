#This is the script for training and testing
# on a model to recognize sentiment in text


#import library
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import numpy as np


#options to make
training_size = 18700 #normally 70% of the dataset
max_length = 30 # the maximal length of sequences of sentencea in the texts


#import data
# reference: rishabhmisra.github.io/publications/
def parse_data(file):
    for l in open(file,'r'):
       yield json.loads(l)

data = list(parse_data('Sarcasm_Headlines_Dataset.json'))

sentences = []
labels = []
urls = []
for item in data:
   sentences.append(item['headline'])
   labels.append(item['is_sarcastic'])
   urls.append(item['article_link'])


#split up training and testing datasets
sentences_training = sentences[0:training_size]
sentences_testing = sentences[training_size:]
labels_training = labels[0:training_size]
labels_testing = labels[training_size:]

#preprocessing on the texts
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences_training)
word_index = tokenizer.word_index

sequences_training = tokenizer.texts_to_sequences(sentences_training)
padded_training = pad_sequences(sequences_training, padding = 'post', maxlen=max_length, truncating='post')
print(padded_training.shape)

sequences_testing = tokenizer.texts_to_sequences(sentences_testing)
padded_testing = pad_sequences(sequences_testing, padding = 'post', maxlen=max_length,truncating='post')
print(padded_testing.shape)

#neural network
model = Sequential([
    #top layer embedding: to get the embedded vectors for each word
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    GlobalAveragePooling1D(),
    # Dense layer: output = activation(dot(input, kernel) + bias)
    Dense(24, activation='relu'), #output arrays of shape(None, 24)
    Dense(1, activation='sigmoid') #output arrays of shape(None, 1)
])
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

/3