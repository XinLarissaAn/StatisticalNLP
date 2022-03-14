#This is the script for training and testing on a model to recognize sentiment in text
#scripter: Xin An

#import library
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#import data