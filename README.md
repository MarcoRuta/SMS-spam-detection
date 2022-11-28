# SMS-spam-detection

## Introduction
Project for the Artificial Intelligence for Cybersecurity course of the master degree in cybersecurity of UniPisa.

The goal is to create a SMS spam detector algorithm using three different methods for processing the text and comparing the results. The methods used are the following:

* Blacklist 
* Locality sensitive hashing
* Countvectorizer + NaiveBayes


The dataset used is te following: https://archive.ics.uci.edu/ml/datasets/spambase which is composed as integration of several pre-existing dataset about SMS spam. The dataset is raw, it has 5 features but only two are meaningful:
- v1: label {"spam","ham"} indicating if the message is spam or not
- v2: raw textual body of the SMS

## Project structure

The directories are structured as following:

 * <b>/dataset:</b> in this directory there is the dataset (spam.csv) and the   computed blacklists (blacklist.txt, cleaned_blacklist.txt) file.

 * <b>/utility:</b> in this directory there are several useful scripts for data analysis (data_analysis.py), data pre-processing (data_preprocessing.py), and for training and testing over the dataset with different methods (train_and_test.py). There is also present a stemming script (stemmer.py), useful for the pre-processing of data.

 * <b>blacklist.py:</b> this script compute a blacklist  and a cleaned blacklist as blacklist and labels the messages as spam/ham based on the presence of blacklisted words.

 * <b>locality_sensitive_hashing.py:</b> this script create and populate an LSH matcher with the spam messages that is used for labeling the messages as spam/ham based on the bucket in which a message will be placed.

 * <b>count_vectorizer.py:</b> this script transform the body of the message in a sparse matrix of tokens count that will be used for training and testing phase of classic ML algorithms.

 * <b>run.sh:</b> shell script that run all the python scripts

 
 
