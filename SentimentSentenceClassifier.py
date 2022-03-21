#This is the script for training and testing
# on a model to recognize sentiment in text


#import library
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import numpy as np

#fix rubuild TensorFlow with te appropraite cmpiler flags error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#options to make
training_size = 19000 #normally 70% of the dataset
max_length = 100 # the maximal length of sequences of sentence in the texts
vocab_size = 20000
embedding_dim = 16

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
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>") #it is important to set num_words
tokenizer.fit_on_texts(sentences_training)
word_index = tokenizer.word_index

sequences_training = tokenizer.texts_to_sequences(sentences_training)
padded_training = pad_sequences(sequences_training, padding = 'post', maxlen=max_length, truncating='post')
print(padded_training.shape)

sequences_testing = tokenizer.texts_to_sequences(sentences_testing)
padded_testing = pad_sequences(sequences_testing, padding = 'post', maxlen=max_length,truncating='post')
print(padded_testing.shape)

#neural network
model = tf.keras.Sequential([
    #top layer embedding: to get the embedded vectors for each word
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    # Dense layer: output = activation(dot(input, kernel) + bias)
    tf.keras.layers.Dense(24, activation='relu'), #output arrays of shape(None, 24)
    tf.keras.layers.Dense(1, activation='sigmoid') #output arrays of shape(None, 1)
])
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()


#traing:fit the model to the data
num_epochs = 30;
#data need to be in the form of an array
padded_training = np.array(padded_training)
labels_training = np.array(labels_training)
padded_testing = np.array(padded_testing)
labels_testing = np.array(labels_testing)

hisory = model.fit(padded_training, labels_training, epochs = num_epochs,
                   validation_data=(padded_testing, labels_testing), verbose=2)

sentence = [
    "granny starting to fear spiders in the garden might be real",
    "the weather today is bright and sunny",
    "I don't want to go out with you",
    "You are taller than a tree"]
sequences = tokenizer.texts_to_sequences(sentence)

padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
print(model.predict(padded))