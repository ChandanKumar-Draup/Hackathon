from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
from flask_assets import Environment, Bundle
from flaskext.mysql import MySQL
from flask_cors import cross_origin
import csv
import io
import json
import os
import time
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


#print(sys.path)

#from output import response
#from models.logistic_regression import LogisticRegression
from models.sklearn_models.logistic_regression import Logistic_Regression
import pandas as pd



class Flask_Object:

    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


flask_object = Flask_Object()


@app.route('/')
def index():
    return 'Hello, World! Welcome to Machine Learning at Click'



def save_dataframe_csv(df, path):
    df.to_csv(path, sep='\t', encoding='utf-8', index=False)



@app.route('/postData', methods=["POST"])
def postData():
    args = request.args
    filename = args.get("fileName")
    f = request.files['fileName']
    print("file", f)
    #print("fileName", filename)
    reading_csv = pd.read_csv(f)
    # reading_csv.to_csv("../data_storage/post_csv_file.csv", sep='\t', encoding='utf-8', index=False)
    save_dataframe_csv(reading_csv, "../data_storage/post_csv_file.csv")
    #output = list(reading_csv.columns.values) if reading_csv is not None else []
    output = reading_csv.head(10).to_html() if reading_csv is not None else ""
    list_names = list(reading_csv.columns.values) if output != "" else []
    file_data = {"data" : output, "headers_list" : list_names}
    return jsonify(file_data)#, jsonfiy()
    #return #jsonify(output)#file_names



# @app.route('/displayData', methods=["GET", "POST"])
# def displayData():
#     args = request.args
#     filename = args.get("fileName")
#     output = pd.read_csv("../data_storage/%s" % filename).head(10).to_html() if filename else ""
#     return output#jsonify(output)
#
#
# @app.route('/displayHeaders', methods=["GET"])
# def displayHeaders():
#     args = request.args
#     filename = args.get("fileName")
#     output_headers = list(pd.read_csv("../data_storage/%s" % filename).columns.values) if filename else []
#     return jsonify(output_headers)
#


import numpy as np



@app.route('/model_fit', methods=["GET"])
def model_fit():
    args = request.args
    print("args", args)
    dataframe = pd.read_csv("../data_storage/post_csv_file.csv", sep='\t', encoding='utf-8')
    print("dataframe", dataframe.head(0))
    all_columns = [column for column in dataframe.columns if column != "Id"]
    X_axis = all_columns[:-1]
    y_axis = all_columns[-1]
    print(all_columns)

    model = Logistic_Regression(dataframe)
    X, y = model.creating_features_target(X_axis, y_axis)
    print("X_head", X.head(10), "y_Head", y.head(10))
    X_train, X_test, y_train, y_test = model.training_testing_split(X, y)

    # X_train.to_csv("../data_storage/data_storage_X_train.csv", sep='\t', encoding='utf-8', index=False)
    # y_train.to_csv("../data_storage/data_storage_y_train.csv", sep='\t', encoding='utf-8', index=False)
    # X_test.to_csv("../data_storage/data_storage_x_test.csv", sep='\t', encoding='utf-8', index=False)
    # y_test.to_csv("../data_storage/data_storage_y_test.csv", sep='\t', encoding='utf-8', index=False)
    save_dataframe_csv(X_train, "../data_storage/data_storage_X_train.csv")
    save_dataframe_csv(y_train, "../data_storage/data_storage_y_train.csv")
    save_dataframe_csv(X_test, "../data_storage/data_storage_x_test.csv")
    save_dataframe_csv(y_test, "../data_storage/data_storage_y_test.csv")



    #print("X_train", X_train.head(10), "y_train", y_train.head(10))
    print("X_train", "y_train")
    fit_result = model.fit_model(X_train, y_train)
    # save the model to disk
    model_fileName = 'finalized_model.sav'
    pickle.dump(model, open(model_fileName, 'wb'))
    return jsonify(fit_result)


@app.route('/model_evaluate', methods=["GET", "POST"])
def model_evaluate():
    print("printlng model")
    # load the model from disk
    model_fileName = 'finalized_model.sav'
    trained_Model = pickle.load(open(model_fileName, 'rb'))
    print(trained_Model)

    X_test = pd.read_csv("../data_storage/data_storage_x_test.csv", sep="\t")
    print(X_test.shape, X_test.head(3))
    y_test = pd.read_csv("../data_storage/data_storage_y_test.csv", sep="\t", names=["Species"])
    print(y_test.shape, X_test.head(3))
    model_class = trained_Model.predict_model(X_test, y_test)
    y_pred = trained_Model.logreg.predict(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(trained_Model.logreg.score(X_test, y_test)))
    print("model_class")

    classification_report = trained_Model.report_classication(y_test, y_pred)
    print(model_class, classification_report)
    print(type(classification_report))

    report_data = []
    lines = classification_report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    print(dataframe.head(2))
    print(dataframe.to_html)
    final_data = {"prediction_result" : model_class, "classfication_report" : dataframe.to_html()}
    return jsonify(final_data)





import argparse
from google.cloud import vision
from google.cloud.vision import types
import io
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']  = "../resources/billflow.json"



@app.route('/detect_labels', methods=["POST"])
def detect_labels():
    """Detects labels in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_label_detection]
    file_image = request.files['fileName']
    print("file_image", file_image)


    # with io.open(file_image, 'rb') as image_file:
    #     content = image_file.read()

    content = file_image.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print('Labels:', labels)

    labels_string = str(labels[0]) if labels is not None else []

    label_1 = labels_string.split("\n")[1]
    label_final = label_1.replace("description: ", "").strip()

    return label_final#jsonify(label_final)











if __name__ == "__main__":
    #app.run(host='0.0.0.0', debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
    #app.run(host='0.0.0.0')
