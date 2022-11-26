from datasketch import MinHash, MinHashLSH
from prettytable import PrettyTable

from utility import data_preprocessing 
from utility import stemmer

X,Y,X_train,X_test,y_train,y_test = data_preprocessing.split_train_test_SMS(0)

spam_messages = []

# grouping up all the spam messages
for text,label in zip(X_test,y_test):
    if(label):
        spam_messages.append(text)

# Initialize MinHashLSH matcher with a Jaccard 
# threshold of 0.5 and 128 MinHash permutation functions
lsh = MinHashLSH(threshold=0.5, num_perm=128)

# Populate the LSH matcher with training spam MinHashes
# Intuition: search similarities with spam 
# Quite a "long" training phase
for text in spam_messages:
    minhash = MinHash(num_perm=128)
    stems = stemmer.stem(text)
    if len(stems) < 2: continue
    for s in stems:
        minhash.update(s.encode('utf-8'))
    # try catch if the same word is inserted several time
    try:
        lsh.insert(text, minhash)
    except ValueError as v:
            continue


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