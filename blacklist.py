# importing all the needed packages
import numpy as np
import nltk.corpus 
from utility import data_preprocessing 
from utility import train_and_test
from utility import stemmer

# downloading a set of worlds from nltk
nltk.download('words')

# retrieving the training and testing sets
X,Y,X_train,X_test,y_train,y_test = data_preprocessing.split_train_test_SMS(0)

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

# building the blacklist as difference between spam and ham words set
blacklist = spam_words - ham_words
print('\nblacklist of {} tokens successfully built!\n'.format(len(blacklist)))

# saving the blacklist
with open('dataset/blacklist.txt','w') as f:
    f.write(str(blacklist))

# evaluating the blacklist
train_and_test.evaluate_blacklist(X_test,y_test,blacklist)

# keeping only meaningful words in the blacklist via an intersection with kltk words set
blacklist = set(nltk.corpus.words.words()).intersection(blacklist)
print('\nThe blacklist was cleaned from strange strings, now is made of {} tokens\n'.format(len(blacklist)))

#saving the new blacklist
with open('dataset/cleaned_blacklist.txt','w') as f:
    f.write(str(blacklist))

# evaluating the blackist
train_and_test.evaluate_blacklist(X_test,y_test,blacklist)