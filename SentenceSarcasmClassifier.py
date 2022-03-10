# This is a test for several word sense disambiguation methods.
# owner: Xin An

#import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

#import data
sentences =[
    'I sit in an office',
    'My colleague also sits in an office'
]

#create an instance of Tokenizer object with top 100 frequent words
tokenizer = Tokenizer(num_words = 100)

#fit the tokenizer on data
tokenizer.fit_on_texts(sentences)

#find the word index and print
word_index = tokenizer.word_index
print(word_index)

#change the texts to sequences and print sequences
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

#use the tokenizer to test on some test data
#which contains words that don't appear in fitting tokenizer
test_data = [
    'Where is my colleague now',
    'Two people sit in an office'
]
sequences_test = tokenizer.texts_to_sequences(test_data)
print(sequences_test)

#method to avoid empty word index in sequences of test data
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
#repeat previous steps
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
sequences_test = tokenizer.texts_to_sequences(test_data)
print(sequences)
print(word_index)
print(sequences_test)
