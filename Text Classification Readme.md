Text Classification and Vector Visualization using TF-IDF and Neural Networks

This project deals with text classification using Multinomial Naive Bayes and Neural Networks, as well as vector visualization using T-SNE. The main focus of this project is the implementation of TF-IDF Vectorization from scratch and its application in various tasks.

Table of Contents

Dataset Overview

Extracting Features from the Dataset

Vector Visualization

Building Neural Networks

Dataset Overview

The project uses two datasets:

20 News Group Dataset

Contains around 18,000 newsgroup posts on 20 topics, split into training and testing subsets.

Dataset link: http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups

Can be imported from sklearn.datasets.

Emotions Dataset

Contains text data classified into different emotions, such as happiness, sadness, anger, etc.

Provided as train.txt and val.txt files.

Extracting Features from the Dataset

To perform machine learning on text documents, we first need to convert the text content into numerical feature vectors. We use TF-IDF Vectorization for this purpose.

TF-IDF Vectorization

Term Frequency-Inverse Document Frequency (TF-IDF) gives a higher weight to words that occur less frequently. It is calculated as follows:

TF-IDF = Term Frequency (TF) \* Inverse Document Frequency (IDF)

idf(t) = log(N/(df + 1))

Vector Visualization

In this unsupervised learning task, we cluster Wikipedia articles into groups using T-SNE visualization after vectorization.

Collecting Articles from Wikipedia

Download articles from Wikipedia, either randomly or based on related topics. You may also use any other data source of your choice.

Cleaning the Data

Clean the data by removing punctuation characters, digits, and upper case letters. This improves the uniformity of the data and enhances clustering performance.

Vectorizing the Articles

Use TfidfVectorizer() or CountVectorizer() from the sklearn library to vectorize the text data.

Plotting Articles

Visualize the groups of articles using T-SNE from the sklearn library. Annotate the points with different markers for different expected groups.

Building Neural Networks

In this task, we use the Emotions Dataset to classify the given text into different emotions like happy, sad, anger, etc. We build neural networks using TensorFlow Keras and evaluate the model on different metrics.

Instructions

Build the neural network model using TensorFlow Keras.

Train and validate the model on the TF-IDF vectors of the text for 10 epochs.

Adjust the batch size according to your computation power (suggested: 8).

Evaluate the model on different metrics and provide observations.

Requirements

To run the project, ensure you have the following packages installed:

numpy

pandas

sklearn

tensorflow

matplotlib

seaborn

wikipedia-api

Getting Started

Clone the repository or download the project files.

Install the required packages using pip.

Open the project in your preferred Python environment and run the code.


