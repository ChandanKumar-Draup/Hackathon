import pandas as pd
import numpy as np
from sklearn import preprocessing
# import matplotlib.pyplot as plt
# plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
# import seaborn as sns
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
from sklearn import metrics
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class Logistic_Regression:

    def __init__(self, dataframe = None):
        self.dataframe = dataframe
        self.logreg = LogisticRegression()
        self.y_pred = None


    def creating_features_target(self, x_cols, y_col):
        file_read = self.dataframe
        X = file_read[x_cols]#file_read[[x_cols]]
        print(x_cols, y_col)
        print(X.head(2))
        try:
            #y = float(y_col)
            y = file_read[float(y_col)]
            print("y_floating output", y)
        except:
            y = file_read[y_col]
            file_read[y_col] = pd.Categorical(file_read[y_col])
            file_read['code'] = file_read[y_col].cat.codes
            y = file_read["code"]
            print("y_file_read", y.head(2))
        return X, y


    def training_testing_split(self, X, y, test_size=0.3, random_state=0):
        print("X_Shape", X.shape, "y_shape", y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("X_Shape", X.shape, "y_shape", y.shape)
        return X_train, X_test, y_train, y_test

    def fit_model(self, X_train, y_train):
        print("X_train", X_train.shape, "y_train", y_train.shape)
        self.logreg.fit(X_train, y_train)
        return "Model Training completed !"
        print("fit model")


    def predict_model(self, X_test, y_test):
        y_pred = self.logreg.predict(X_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(self.logreg.score(X_test, y_test)))
        return 'Ac`curacy of logistic regression classifier on test set: {:.2f}'.format(self.logreg.score(X_test, y_test))


    ### confusion_matrix
    def confusion_matrix(self, y_test, y_pred):
        confusion_matrix = confusion_matrix(y_test, y_pred)
        print(confusion_matrix)
        return confusion_matrix

    ### classification_report
    def report_classication(self, y_test, y_pred):
        print(classification_report(y_test, y_pred))
        return classification_report(y_test, y_pred)
