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

 
 
## Results

Here are listed all the confusion matrix and metrics obtained from the different approaches. The matrix was calculated over a 30% testing set. For the CountVectorizer approach was also used k-fold validation with k=10.
<br><br><br>


<center> <h1>Blacklist results</h1> </center>

<p align="center">
<table>
    <thead>
        <tr>
            <th>Confusion Matrix</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>[1421&nbsp&nbsp&nbsp&nbsp27]<br>[63&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp161]</td>
            <td>0.946</td>
            <td>0.856</td>
            <td>0.719</td>
            <td>0.782</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>


<center> <h1>Cleaned Blacklist results</h1> </center>

<p align="center">
<table>
    <thead>
        <tr>
            <th>Confusion Matrix</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>[1425&nbsp&nbsp&nbsp&nbsp23]<br>[154&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp70]</td>
            <td>0.894</td>
            <td>0.753</td>
            <td>0.312</td>
            <td>0.441</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>


<center> <h1>LSH results</h1> </center>

<p align="center">
<table>
    <thead>
        <tr>
            <th>Confusion Matrix</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>[1429&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp5]<br> [100&nbsp&nbsp&nbsp&nbsp123]</td>
            <td>0.937</td>
            <td>0.961</td>
            <td>0.552</td>
            <td>0.701</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>


<center> <h1>Count Vectorizer results</h1> </center>


<p align="center">
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Confusion Matrix</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Logistic Regression</td>
            <td><b>[1447&nbsp&nbsp&nbsp&nbsp&nbsp1]<br> [40&nbsp&nbsp&nbsp&nbsp&nbsp184]</td>
            <td>0.975</td>
            <td>0.994</td>
            <td>0.821</td>
            <td>0.899</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>Decision Tree</td>
            <td><b>[1423&nbsp&nbsp&nbsp25]<br> [108&nbsp&nbsp&nbsp116]</td>
            <td>0.920</td>
            <td>0.822</td>
            <td>0.517</td>
            <td>0.635</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>KNeighbors</td>
            <td><b>[1448&nbsp&nbsp&nbsp&nbsp&nbsp0]<br> [135&nbsp&nbsp&nbsp&nbsp&nbsp89]</td>
            <td>0.919</td>
            <td>1.0</td>
            <td>0.397</td>
            <td>0.568</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>SVC</td>
            <td><b>[1448&nbsp&nbsp&nbsp&nbsp&nbsp0]<br> [43&nbsp&nbsp&nbsp&nbsp&nbsp181]</td>
            <td>0.974</td>
            <td>1.0</td>
            <td>0.808</td>
            <td>0.893</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>Naive Bayes</td>
            <td><b>[1436&nbsp&nbsp&nbsp12]<br> [18&nbsp&nbsp&nbsp&nbsp&nbsp206]</td>
            <td>0.982</td>
            <td>0.944</td>
            <td>0.919</td>
            <td>0.932</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>

<center> <h2>K-fold cross validation</h2> </center>
<p align="center">
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th> AVG Accuracy</th>
            <th>AVG Precision</th>
            <th>AVG Recall</th>
            <th>AVG F1</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Logistic Regression</td>
            <td>0.98</td>
            <td>0.986</td>
            <td>0.927</td>
            <td>0.954</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>Decision Tree</td>
            <td>0.933</td>
            <td>0.94</td>
            <td>0.761</td>
            <td>0.819</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>KNeighbors</td>
            <td>0.931</td>
            <td>0.962</td>
            <td>0.745</td>
            <td>0.809</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>SVC</td>
            <td>0.978</td>
            <td>0.987</td>
            <td>0.92</td>
            <td>0.95</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>Naive Bayes</td>
            <td>0.981</td>
            <td>0.953</td>
            <td>0.966</td>
            <td>0.959</td>
        </tr>
        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>
