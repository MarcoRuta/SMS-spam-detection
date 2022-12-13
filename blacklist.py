# importing all the needed packages
import numpy as np
import nltk.corpus 
from utility import data_cleaning 
from utility import stemmer
from prettytable import PrettyTable

# retrieving the data needed (cleaned)
db,X,Y,X_train,X_test,y_train,y_test = data_cleaning.get_data()

# Function that evaluate the metrics of a given blacklist over a given testing set
def evaluate_blacklist(X_test,y_test,blacklist):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for text,label in zip(X_test['sms'],y_test):
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

    t.add_row([str(confusion_matrix[0:1])+"\n"+str(confusion_matrix[1:2]), accuracy, precision, recall, f1 
            ])

    t.add_row( ['', '', '', '', ''] )
    print( t )

# downloading a set of worlds from nltk
nltk.download('words')

# set of the spam words
spam_words = set()
# set of legitimate words
ham_words = set()

for text,label in zip(X_train['sms'],y_train):
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
evaluate_blacklist(X_test,y_test,blacklist)

# keeping only meaningful words in the blacklist via an intersection with kltk words set
blacklist = set(nltk.corpus.words.words()).intersection(blacklist)
print('\nThe blacklist was cleaned from strange strings, now is made of {} tokens\n'.format(len(blacklist)))

#saving the new blacklist
with open('dataset/cleaned_blacklist.txt','w') as f:
    f.write(str(blacklist))

# evaluating the blackist
evaluate_blacklist(X_test,y_test,blacklist)