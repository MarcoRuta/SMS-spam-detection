from data_preprocessing import split_train_test_SMS
from train_and_test import classify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


#@param 1 if you want to see a graphical representation of the dataset, 0 otherwise
X_train,X_test,y_train,y_test = split_train_test_SMS(1)

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)
# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


classify(training_data,y_train,testing_data,y_test)



