# Databricks notebook source
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

# COMMAND ----------

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

# COMMAND ----------

def appl_func(fun):
  for i in header_lst:
    train_file[str(i)] = train_file[str(i)].apply(lambda x: fun(x))
  return train_file

# COMMAND ----------

train_file = pd.read_csv("/dbfs"+path)

# COMMAND ----------

type(train_file.apply(lambda col: col.str.strip()))
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.sparse as sp

def vecto(vector, file): 
  if vector == "CountVectorizer": 
    vect = CountVectorizer()
    train = sp.hstack(file.apply(lambda col: vect.fit_transform(col)))
    return train
  elif vector == "CountVectorizer": 
    vect = TfidfVectorizer()
    train = sp.hstack(file.apply(lambda col: vect.fit_transform(col)))
    return train


# COMMAND ----------

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import metrics
def mod(model):
  if model == 'RandomForestClassifier':
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=500, class_weight = 'balanced')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("accuracy:",metrics.accuracy_score(y_test, predictions))
    return clf
  elif model == 'Support Vector Machines(rbf classification)':
    clf = svm.SVC()
    clf.fit(X_train, y_train) 
    predictions = clf.predict(X_test)
    print("accuracy:",metrics.accuracy_score(y_test, predictions))
    return clf
  elif model == 'Support Vector Machines(Multi-class classification)':
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("accuracy:",metrics.accuracy_score(y_test, predictions))
    return clf
  elif model == 'LogisticRegression':
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("accuracy:",metrics.accuracy_score(y_test, predictions))
    return clf
  elif model == 'Neural Networks(MLPClassifier)':
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("accuracy:",metrics.accuracy_score(y_test, predictions))
    return clf
  else:
    print("Please Select the Model Listed Below:")

# COMMAND ----------

#Image Recognition

# COMMAND ----------

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

# COMMAND ----------

images = []
im_mn = []
import cv2
import os
def load_images_from_folder(folder):    
    for filename in os.listdir(folder):
        print(filename)
        for i in os.listdir('/dbfs/FileStore/tables/data1/natural_images/'+filename):
#             print(i)
            if i.endswith(".jpg"):
                img = cv2.imread(os.path.join('/dbfs/FileStore/tables/data1/natural_images/'+filename, i))
                img = cv2.resize(img, (128, 128))
#     img = img_to_array(img)
                im_mn_x = str(filename)
            im_mn.append(im_mn_x)
            images.append(img)
    return images
    return im_mn

# COMMAND ----------

img_load_arr=np.array(img_load)
img_load_arr_resized = img_load_arr.astype('float32')

from sklearn.preprocessing import LabelEncoder
im_mn_taggs = LabelEncoder().fit(im_mn)
encoded = im_mn_taggs.transform(im_mn)
encoded_labels = np.array(encoded, dtype=float)


from keras.utils import np_utils
def one_hot_encode(y):

    # one hot encode outputs
    y = np_utils.to_categorical(y)
    num_classes = y.shape[1]
    return y
  
ec1 = one_hot_encode(ec)  


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(img_load_arr_resized, ec1, test_size=.20, random_state=0)

# COMMAND ----------

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras import losses
from keras import optimizers
 
def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(ec)), activation='softmax'))
     
    return model

# COMMAND ----------

model1 = createModel()
batch_size = 256
epochs = 100
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 
history = model1.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(X_test, y_test))
 
model1.evaluate(X_test, y_test)