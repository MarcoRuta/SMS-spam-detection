from utility import data_preprocessing 
from utility import train_and_test
from sklearn.feature_extraction.text import CountVectorizer
import sys



#@param 1 if you want to see a graphical representation of the dataset, 0 otherwise
X,Y,X_train,X_test,y_train,y_test = data_preprocessing .split_train_test_SMS(0)


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
    training_data = vectorizer.fit_transform(X_train)
    testing_data = vectorizer.transform(X_test)

    full_data = vectorizer.transform(X)

    print("\n\n"+"*"*60+"\nResults for vectorizer: "+features+"\n"+"*"*60+"\n\n")
    train_and_test.classify(training_data,y_train,testing_data,y_test)

full_data = count_vectorizers[-1].transform(X)
train_and_test.k_fold_cross_validation(full_data,Y,8)
 








