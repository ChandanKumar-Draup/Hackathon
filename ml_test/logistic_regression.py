# import numpy as np
# import pandas as pd
#
# class LogisticRegression:
#     def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
#         self.lr = lr
#         self.num_iter = num_iter
#         self.fit_intercept = fit_intercept
#         self.verbose = True
#
#     def __add_intercept(self, X):
#         intercept = np.ones((X.shape[0], 1))
#         print("intercept", intercept.shape)
#         print(X.shape[0])
#         print(X.shape)
#         return np.concatenate((intercept, X), axis=1)
#
#     def __sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))
#
#     def __loss(self, h, y):
#         return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
#
#     def fit(self, X, y):
#         if self.fit_intercept:
#             X = self.__add_intercept(X)
#
#         # weights initialization
#         self.theta = np.zeros(X.shape[1])
#
#         for i in range(self.num_iter):
#             z = np.dot(X, self.theta)
#             h = self.__sigmoid(z)
#             gradient = np.dot(X.T, (h - y)) / y.size
#             self.theta -= self.lr * gradient
#
#             if(self.verbose == True and i % 10000 == 0):
#                 z = np.dot(X, self.theta)
#                 h = self.__sigmoid(z)
#                 print(f'loss: {self.__loss(h, y)} \t')
#                 return f'loss: {self.__loss(h, y)} \t'
#
#     def predict_prob(self, X):
#         if self.fit_intercept:
#             X = self.__add_intercept(X)
#
#         return self.__sigmoid(np.dot(X, self.theta))
#
#     def predict(self, X, threshold):
#         return self.predict_prob(X) >= threshold
#
#
#
#
#
#
#
# ###### checking the values
#
# args = {"fileName" : "Iris.csv", "X_axis" : "SepalLengthCm", "y_axis" : "Species"}
# print("args", args)
# filename = args.get("fileName")
# default_param = {"learning_rate" : 0.01, "num_iter": 100000, "fit_intercept" : True, "verbose": False}
# learning_rate = args.get("learning_rate") if args.get("learning_rate") else  default_param.get("learning_rate")
# num_iter = args.get("num_iter") if args.get("num_iter") else default_param.get("num_iter")
# fit_intercept = args.get("fit_intercept") if args.get("fit_intercept") else default_param.get("fit_intercept")
# verbose = args.get("verbose") if args.get("verbose") else default_param.get("verbose")
# X_axis = args.get("X_axis")
# y_axis = args.get("y_axis")
# file_read = pd.read_csv("../data_storage/%s" % filename)
# X = file_read[[X_axis]]
# print(X_axis, y_axis)
# print(X.head(2))
#
# # y = file_read[y_axis]
# file_read[y_axis] = pd.Categorical(file_read[y_axis])
# file_read['code'] = file_read[y_axis].cat.codes
# y = file_read["code"]
#
# print(y.head(3))
#
# # model = LogisticRegression(learning_rate, num_iter, fit_intercept, True)
# # fit_result = model.fit(X, y)
# # print("fit result")
#
#
#
# #####
#
# import pandas as pd
# import numpy as np
# from sklearn import preprocessing
# # import matplotlib.pyplot as plt
# # plt.rc("font", size=14)
# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
# # import seaborn as sns
# # sns.set(style="white")
# # sns.set(style="whitegrid", color_codes=True)
# from sklearn import metrics
# import sklearn
# #
# # #iris = sklearn.datasets.load_iris()
# # iris = file_read
# # X = file_read[["SepalLengthCm"]]#, "SepalWidthCm"]]# + file_read["SepalWidthCm"]#iris.data[:, :2]
# # y = file_read["Species"]#(iris.target != 0) * 1
# #
# # print(X.head(3))
# # print(y.head(3))
# #
# #
# #
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# print("fit model")
# #
# y_pred = logreg.predict(X_test)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#
#
# ### confusion_matrix
#
# from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print(confusion_matrix)
#
# #### classification report
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))



import argparse
from google.cloud import vision
from google.cloud.vision import types
import io
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']  = "resources/billflow.json"
def detect_labels(path):
    """Detects labels in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_label_detection]
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
#     print('Labels:')

    return labels[0]


print(detect_labels("resources/a3c.jpg"))
