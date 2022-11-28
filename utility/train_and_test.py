import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from prettytable import PrettyTable
from sklearn.model_selection import cross_val_score, cross_val_predict
from termcolor import colored
from utility import stemmer
from datasketch import MinHash, MinHashLSH

# Classifiers that will be used
classifiers = [
    LogisticRegression( max_iter=500, solver="lbfgs" ),
    DecisionTreeClassifier(max_depth = 5),
    KNeighborsClassifier( 3 ),
    SVC(),
    MultinomialNB()
]

# Names of the classifiers
names = [
    'Logistic Regression',
    'Decision Tree',
    'KNeighbors',
    'SVC',
    'Naive Bayes'
]


# Function that train and test all the classifiers chosen and return the results
# in a fancy table
def classify(train_x, train_y, test_x, test_y):

    t = PrettyTable(
        ['Name', 'Confusion Matrix', 'Accuracy', 'Precision', 'Recall', 'F1'] )

    for name, clf in zip( names, classifiers ):

        # fitting the classifier
        clf.fit( train_x, train_y )
        # testing the classifier over the testing set
        predictions = clf.predict( test_x )
        # computing metrics over the testing set classification
        _accuracy = format(accuracy_score(test_y,predictions))
        _precision = format(precision_score(test_y,predictions))
        _recall = format(recall_score(test_y,predictions))
        _f1 = format(f1_score(test_y,predictions))
        _matrix = confusion_matrix(test_y,predictions)

        t.add_row(
            [colored( name, 'blue' ), _matrix, _accuracy[0:5], _precision[0:5], _recall[0:5], _f1[0:5] 
            ])

        t.add_row( ['', '', '', '', '', ''] )
    print( t )


# Function that train and test the classifier with k-fold cross validation and return
# the results in a fancy table
def k_fold_cross_validation(x,y,k):

    t = PrettyTable(
        ['Name', ' AVG Accuracy', 'AVG Precision', 'AVG Recall', 'AVG F1'] )

    for name, clf in zip( names, classifiers ):

        _avg_accuracy = cross_val_score( clf, x, y, cv=k, scoring='accuracy' )
        _avg_precision = cross_val_score( clf, x, y, cv=k, scoring='precision_macro' )
        _avg_recall = cross_val_score( clf, x, y, cv=k, scoring='recall_macro' )
        _avg_F1 = cross_val_score( clf, x, y, cv=k, scoring='f1_macro' )

        t.add_row(
            [colored( name, 'blue' ), round(_avg_accuracy.mean(),3), round(_avg_precision.mean(), 3), round(_avg_recall.mean(), 3), round(_avg_F1.mean(), 3)
            ])

        t.add_row( ['', '', '', '', ''] )
    print(t)
    return

# Function that evaluate the metrics of a given blacklist over a given testing set
def evaluate_blacklist(X_test,y_test,blacklist):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for text,label in zip(X_test,y_test):
        stems = set(text.split())
        stems_set = set(stems)
        # email's words are in blacklist
        if stems_set & blacklist: 
            if label:
                tp += 1
            else:
                fp += 1 
        # email's words are not in blacklist
        else: 
            if label:
                fn += 1
            else:
                tn += 1


    confusion_matrix = [[tn,fp],
                        [fn,tp]]


    accuracy = round((tp+tn)/(tp + fp + tn + fn), 3)
    precision = round((tp /(tp + fp)), 3)
    recall = round((tp / (tp + fn)), 3)
    f1 = round(((2*precision*recall)/(precision + recall)), 3)

    t = PrettyTable(
            ['Confusion Matrix', 'Accuracy', 'Precision', 'Recall', 'F1'] )

    t.add_row([confusion_matrix, accuracy, precision, recall, f1 
            ])

    t.add_row( ['', '', '', '', ''] )
    print( t )

# Function that evaluate the metrics of a given lsh on a given testing set
def lsh_classify(X_test,y_test,lsh):

    def lsh_predict_label(stems):

        minhash = MinHash(num_perm=128)
        if len(stems) < 2:
            return -1
        for s in stems:
            minhash.update(s.encode('utf-8'))
        matches = lsh.query(minhash)
        if matches:
            return 1
        else:
            return 0


    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for text,label in zip(X_test,y_test):
        stems = stemmer.stem(text)
        pred = lsh_predict_label(stems)
        # parsing error
        if pred == -1:
            continue
        # predicted spam
        elif pred == 1: 
            if label:
                tp += 1
            else:
                fp += 1 
        # predicted ham
        elif pred == 0: 
            if label == 1:
                fn += 1
            else:
                tn += 1

    confusion_matrix = [[tn,fp],
                        [fn,tp]]


    accuracy = round((tp+tn)/(tp + fp + tn + fn), 3)
    precision = round((tp /(tp + fp)), 3)
    recall = round((tp / (tp + fn)), 3)
    f1 = round(((2*precision*recall)/(precision + recall)), 3)

    t = PrettyTable(
            ['Confusion Matrix', 'Accuracy', 'Precision', 'Recall', 'F1'] )

    t.add_row([confusion_matrix, accuracy, precision, recall, f1 
            ])

    t.add_row( ['', '', '', '', ''] )
    print( t )