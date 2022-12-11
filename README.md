# SMS-spam-detection

## Introduction
Project for the Artificial Intelligence for Cybersecurity course of the master degree in cybersecurity of UniPisa.

The goal is to create a SMS spam detector system using three different approaches for processing the text and comparing the obtained results. The methods used are the following:

* Blacklist 
* Locality sensitive hashing
* Countvectorizer + ML algorithms


The dataset used is the following: https://archive.ics.uci.edu/ml/datasets/spambase which is composed as integration of several pre-existing dataset about SMS spam. It is one of the best collection of SMS messages aviable for free because of the privacy concerns of this topic. The dataset is raw, it has 5 features but only two are meaningful:
- <b>v1</b>: label {"spam","ham"} indicating if the message is spam or legit
- <b>v2</b>: raw textual body of the SMS

## Project structure

The directories are structured as following:

 * <b>/dataset:</b> in this directory there is the dataset (spam.csv) and the computed blacklists (blacklist.txt, cleaned_blacklist.txt) files.

 * <b>/utility:</b> in this directory there are several useful scripts for data analysis (data_analysis.py), data pre-processing (data_cleaning.py), and there is also a stemming script (stemmer.py), useful for the pre-processing of textual data.

 * <b>blacklist.py:</b> this script compute a blacklist and a cleaned blacklist (without meaningless strings) and labels the messages as spam/ham based on the presence of blacklisted words.

 * <b>locality_sensitive_hashing.py:</b> this script create and populate an LSH matcher with the spam messages. This is used for labeling the messages as spam/ham based on the bucket in which a message will be placed by a MinHash algorithm.

 * <b>count_vectorizer.py:</b> this script transform the body of the message in a sparse matrix of tokens count that will be used for training and testing phase of classic ML algorithms as Naive Bayes and SVM (The selection of this algorithms is based on literature evidence).

 * <b>UI.py:</b> this script deploy a mock-up web app that uses the pretrained ML algorithms to predict the label of a user input. 

 * <b>run.sh:</b> shell script useful to run all the project 

