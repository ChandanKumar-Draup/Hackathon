import pandas as pd
import pandas as pd
import numpy as np
import collections
import re
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()
import unidecode


class Controllers:

    def __init__(self, dataframe, model):
        self.dataframe = args
        self.model = None



def clean_lower(x):
  x = x.lower()
  return x


def clean_upper(x):
  x = x.upper
  return x


def clean_stop_words(x):
  stop = stopwords.words('english')
  for m in stop:
      x = x.replace(m, '')
  return x


def rem_numb(x):
  nub = "1 2 3 4 5 6 7 8 9 0"
  nub = nub.split (' ')
  for m in nub:
      x = x.replace(m, '')
  return x


def clean_std(x):
  replacing_words = "( ) , [ ] \ / ! ` ~ ? | - _ = + ; { } > < @ # $ % ^ & * ' ."
  replacing_words = replacing_words.split (' ')
  for m in replacing_words:
      x = x.replace(m, '')
  return x


def lemitiz(x):
  lemma = WordNetLemmatizer()
  words = x.split (' ')
  lem_words= [lemma.lemmatize(word) for word in words]
  lem_words = ' '.join(lem_words)
  print(lem_words)


def unicode_dec(x):
  x = unidecode.unidecode(x)
  return x
