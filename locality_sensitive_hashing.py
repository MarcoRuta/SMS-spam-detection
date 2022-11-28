# importing all the needed packages
from datasketch import MinHash, MinHashLSH
from prettytable import PrettyTable
from utility import data_preprocessing 
from utility import stemmer
from utility import train_and_test

# retrieving the training and testing sets
X,Y,X_train,X_test,y_train,y_test = data_preprocessing.split_train_test_SMS(0)

# spam messages list
spam_messages = []

# grouping up all the spam messages
for text,label in zip(X_train,y_train):
    if(label):
        spam_messages.append(text)

# Initialize MinHashLSH matcher with 
# Jaccard threshold of 0.5 
# 128 MinHash permutation functions
lsh = MinHashLSH(threshold=0.5, num_perm=128)

# Populate the LSH matcher with training spam MinHashes
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

# Evaluating the lsh
train_and_test.lsh_classify(X_test,y_test,lsh)