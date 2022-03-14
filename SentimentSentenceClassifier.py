#This is the script for training and testing
# on a model to recognize sentiment in text


#import library
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
