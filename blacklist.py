import numpy as np
import nltk.corpus 
from utility import data_preprocessing 
from utility import train_and_test
from utility import stemmer

nltk.download('words')

X,Y,X_train,X_test,y_train,y_test = data_preprocessing.split_train_test_SMS(0)

# saving the labels and the bodies of the messages in two different files
np.savetxt('dataset/body.txt', X.values, fmt='%s')
np.savetxt('dataset/labels.txt', Y.values, fmt='%s')

bodies_file = 'dataset/body.txt'
labels_file = 'labels.txt'

# set of the spam words
spam_words = set()
# set of legitimate words
ham_words = set()

for text,label in zip(X_train,y_train):
    # retrieve the set of words from the message
    stems = stemmer.stem(text)
    # update the list of spam words
    if(label):
        spam_words.update(stems)
    # update the list of ham words
    else: 
        ham_words.update(stems)

# building the blacklist
blacklist = spam_words - ham_words
print('\nblacklist of {} tokens successfully built!\n'.format(len(blacklist)))

# evaluating the blacklist
train_and_test.evaluate_blacklist(X_test,y_test,blacklist)

# removing all the strange strings from the blacklist
word_set = set(nltk.corpus.words.words())
blacklist = word_set.intersection(blacklist)
print('\nThe blacklist was cleaned from strange strings, now is made of {} tokens\n'.format(len(blacklist)))

# evaluating the blackist
train_and_test.evaluate_blacklist(X_test,y_test,blacklist)















