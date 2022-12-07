import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud, STOPWORDS
import data_preprocessing

# reading the csv file into a pandas dataframe
db = pd.read_csv (r'dataset/spam.csv',encoding='latin-1')

# convert the categorical labels of v1 (ham,spam) in binary (0,1) labels
db[['v1']] = db[['v1']].apply(LabelEncoder().fit_transform)

# labels and features raw arrays
Y = db['v1']
X = db['v2']

# print in the shell some general info about the dataset
def print_info():
    print("Head of the dataset: ")
    print(db.head())
    print("Shape of the dataset: ")
    db.shape
    print("Number of null instances in the dataset: "+str(db.isnull().sum))

# plot a cloudwords with the most frequent words in the spam messages
def cloud_words():
        spam_messages = db[Y==1]['v2']

        words = ""

        for message in spam_messages:
            words += message

        wordcloud = WordCloud(width=800, height=400, max_font_size=100, max_words=50,colormap='summer').generate(words)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

# plot a cloudwords with the most frequent words in the spam messages
def cloud_words_stemmed():
        spam_messages = db[Y==1]['v2']

        words = ""

        for message in spam_messages:
            
            words += message

        stop = set(STOPWORDS)

        wordcloud = WordCloud(width=800, height=400, max_font_size=100,normalize_plurals=True,min_word_length=3, max_words=50,colormap='summer',stopwords=stop).generate(words)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

# print in the shell some general info about the dataset
def print_info():
    print("Head of the dataset: ")
    print(db.head())
    print("Shape of the dataset: ")
    db.shape
    print("Number of null instances in the dataset: "+str(db.isnull().sum))

# print a bar chart that shows the number of words distribution in the dataset
def print_sms_lenght():

    # adding a feature with the number of word in each message
    db['Count']=0
    for i in np.arange(0,len(db.v2)):
        db.loc[i,'Count'] = len(db.loc[i,'v2'])

    # counting up and sort the number of word for ham messages
    ham  = db[Y == 0]
    ham_count  = pd.DataFrame(pd.value_counts(ham['Count'],sort=True).sort_index())

    # counting up and sort the number of word for spam messages
    spam = db[Y == 1]
    spam_count = pd.DataFrame(pd.value_counts(spam['Count'],sort=True).sort_index())

    # plotting the bar chart
    fig, ax = plt.subplots(figsize=(17,5))
    spam_count['Count'].value_counts().sort_index().plot(ax=ax, kind='bar',facecolor='red',label = "spam")
    ham_count['Count'].value_counts().sort_index().plot(ax=ax, kind='bar',facecolor='green',label = "ham")
    plt.suptitle("Distribution of number of words in the messages")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    print_info()
    data_preprocessing.split_train_test_SMS(1)
    print_sms_lenght()
    cloud_words()
    cloud_words_stemmed()