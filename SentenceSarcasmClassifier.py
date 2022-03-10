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

# create an instance of Tokenizer object with top 100 frequent words
tokenizer = Tokenizer(num_words = 100)
#fit the tokenizer on data
tokenizer.fit_on_texts(sentences)
# find the word index and print
word_index = tokenizer.word_index
print(word_index)

