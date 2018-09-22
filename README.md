# Hackathon
Introduction: Machine learning (ML) has achieved considerable successes in recent years and an ever-growing number of disciplines rely on it. However, this success crucially relies on human machine learning experts to perform the following tasks:
	•	Preprocess and clean the data.
	•	Select and construct appropriate features.
	•	Select an appropriate model family.
	•	Optimize model hyperparameters.
	•	Postprocess machine learning models.
	•	Critically analyze the results obtained.

Automated Machine Learning provides methods and processes to make Machine Learning available for non-Machine Learning experts, to improve efficiency of Machine Learning and to accelerate research on Machine Learning. As the complexity of these tasks is often beyond non-ML-experts, the rapid growth of machine learning applications has created a demand for off-the-shelf machine learning methods that can be used easily and without expert knowledge.  
Aim of Project/Problem Statement: The success of machine learning in a broad range of applications has led to an ever-growing demand for machine learning systems that can be used off the shelf by non-experts( people with less programming/ ML experience like consultants). To be effective in practice, such systems need to automatically choose a good algorithm and feature preprocessing steps for a new dataset at hand, and also set their respective hyperparameters. At its core, every effective machine learning service needs to solve the fundamental problems of deciding which machine learning algorithm to use on a given dataset, whether and how to preprocess its features, and how to set all hyperparameters. This is the problem we address in this work. More specifically, we investigate automatic machine learning, the problem of automatically (without human input) producing test set predictions for a new dataset within a fixed computational budget. 

Pros of this model over the ones already flooding the market:
a) cost-effective, less computational budget
b) a generic model for creating basic ML models by less expertise people

Two important challenges in AutoML are that (1) no single machine learning method performs best on all datasets and (2) some machine learning methods (e.g., non-linear SVMs) crucially rely on hyperparameter optimisation. Hence, the use-case can vary hence we choose the most common classifiers model for creating this generic model.

Natural Language for Automatic Machine Learning: 
It discovers syntax, entities, and sentiment in text, and classifies text into a predefined set of categories. If your text consists of news articles or other content you'd like categorized, or if you're interested in discovering the sentiment of your examples, the Natural Language API is worth trying. But if your text examples don't fit neatly into the sentiment-based or vertical-topic-based classification scheme available in the Natural Language API, and you'd like to use your own labels instead, it's worth experimenting with a custom classifier to see if it fits your needs.
Data Preparation :  
Upload data > Type of Dataset (CSV for NLP/jpg file for Image) > import the required file you need. In order to train a custom model with AutoML Natural Language, you will need to supply labeled examples of the kinds of text items (inputs) you would like to classify, and the categories or labels (the answer) you want the ML systems to predict. Dataset contains the training/tetsing/validation data (split the inputs data as per specification by default 80/20%) and some dat pre-processing features (text cleaning, image resize/reshape)
Choose the Model Type:
a) Natural Language Processing: Supervised Classification Models incorporated are Random forest, Support Vector Machines(rbf classification), Support Vector Machines(Multi-class classification), LogisticRegression, Neural Networks(MLPClassifier). Supervised machine learning is where the model is trained by input data and expected output data.
UnSupervised Classification: k-Means, Hierarchical clustering
b) Image Classification: 
Train the model: by finding the X-train and Y-train (axes) and fitting the model on it.
Test the model : After the training of the model, you can assess your custom model's performance using the model's output on test examples, and common machine learning metrics by evaluating the model performance based on the accuracy, precision and recall on the basis of the confusion matrix for a detailed analysis.
Save the Model.
Тo create such model, it is necessary to go through the following phases:
	.	model construction
	.	model trainingAfter model construction it is time for model training. In this phase, the model is trained using training data and expected output for this data.
It’s look this way: model.fit(training_data, expected_output).
Progress is visible on the console when the script runs. At the end it will report the final accuracy of the model.
Once the model has been trained it is possible to carry out model testing. During this phase a second set of data is loaded. This data set has never been seen by the model and therefore it’s true accuracy will be verified.
After the model training is complete, and it is understood that the model shows the right result, it can be saved by: model.save(“name_of_file.h5”).
Finally, the saved model can be used in the real world. The name of this phase is model evaluation. This means that the model can be used to evaluate new data.
	.model testing	.model evaluation
TECHNOLOGIES USED:  Machine learning involves using data to train algorithms to achieve a desired outcome. The specifics of the algorithm and training methods change based on the use case. There are many different subcategories of machine learning, all of which solve different problems and work within different constraints. AutoML Natural Language enables you to perform supervised learning, which involves training a computer to recognize patterns from labeled data. Using supervised learning, we can train a custom model to recognize content that we care about in text.
b) Image Classification: For applying the model to Image classification, it needs to be more tractable to large-scale datasets. Creation of Automatic ML can help in the selection of best layer (which can then be stacked many times in a flexible manner) to create a final network. Input: jpg file, Output: Classes of images (labels), Classification model : CNN, Keras 
Techniques and tools Used in Image classification : 
Python syntax (for this project), Keras framework (high-level neural network API written in Python). Since Keras can’t work by itself, it needs a backend for low-level operations, a dedicated software library — Google’s TensorFlow has been installed. The image is passed through a series of convolutional, nonlinear, pooling layers and fully connected layers, and then generates the output.

Why CNN?? Other learning algorithms or models can also be used for image classification. However CNN has emerged as the model of choice for multiple reasons. These include the multiple uses of the convolution operator in image processing, The CNN architecture implicitly combines the benefits obtained by a standard neural network training with the convolution operation to efficiently classify images. Further, being a neural network, the CNN (and its variants) are also scalable for large datasets, which is often the case when images are to be classified. 

Deploying the Machine Learning generic models in Production as APIs (using Flask) : Combining all the functionalities of a generic model with the Flask API for creating the front-end UI. Flask is very minimal since you only bring in the parts as you need them. To demonstrate this, here’s the Flask code to create a very simple web server. Once executed, you can navigate to the web address, which is shown the terminal, and observe the expected result.
