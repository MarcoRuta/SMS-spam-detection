# importing all the needed packages
from utility import data_preprocessing 
from utility import train_and_test
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from prettytable import PrettyTable
from sklearn.model_selection import cross_val_score
from termcolor import colored


# Classifiers that will be used
classifiers = [
    SGDClassifier(),
    MultinomialNB()
]

# Names of the classifiers
names = [
    'SGD SVM',
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

# param 1 if you want to see a graphical representation of the dataset, 0 otherwise
X,Y,X_train,X_test,y_train,y_test = data_preprocessing .split_train_test_SMS(0)

# trying several count_vectorizer the best results are with the last one
count_vectorizers = [
    #CountVectorizer(stop_words='english'),
    #CountVectorizer(stop_words='english', lowercase=True),
    #CountVectorizer(stop_words='english', strip_accents='unicode'),
    #CountVectorizer(strip_accents='unicode', lowercase=True),
    CountVectorizer(lowercase=True,stop_words='english',strip_accents='unicode')
]

vectorizer_features = [
    #"stopwords",
    #"stopwords && lowercase",
    #"stopwords && accents",
    #"accents && lowercase",
    "stopwords && accents && lowercase"
]

for vectorizer,features in zip(count_vectorizers, vectorizer_features):
    # transforming the data in a sparse matrix of tokens count
    training_data = vectorizer.fit_transform(X_train)
    testing_data = vectorizer.transform(X_test)

    # transformin also the full dataset for k-cross validation
    full_data = vectorizer.transform(X)

    # classify with 70 % training and 30% testing
    print("\n\n"+"*"*60+"\nResults for vectorizer: "+features+"\n"+"*"*60+"\n\n")
    train_and_test.classify(training_data,y_train,testing_data,y_test)

# classify with k-cross fold validation (k = 10)
full_data = count_vectorizers[-1].transform(X)
train_and_test.k_fold_cross_validation(full_data,Y,8)