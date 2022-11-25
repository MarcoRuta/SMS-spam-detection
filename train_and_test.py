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

# Classifiers that will be used
classifiers = [
    LogisticRegression( max_iter=500, solver="lbfgs" ),
    DecisionTreeClassifier(max_depth = 5),
    KNeighborsClassifier( 3 ),
    GradientBoostingClassifier( n_estimators=10 ),
    SVC(),
    MultinomialNB()
]

# Names of the classifiers
names = [
    'Logistic Regression',
    'Decision Tree',
    'KNeighbors',
    'Gradient Boosting',
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
            [colored( name, 'blue' ), _matrix, _accuracy, _precision, _recall, _f1 
            ])

        t.add_row( ['', '', '', '', '', ''] )
    print( t )
