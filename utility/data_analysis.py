import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import data_cleaning
import stemmer
import seaborn as sns

# retrieving the data needed (cleaned)
db,X,Y,X_train,X_test,y_train,y_test = data_cleaning.get_data()

# check of the text conversion algorithm
def text_conversion_test():
    print(db.sms.values[53])
    print(stemmer.stem(db.sms.values[53])) 
    cv = CountVectorizer(lowercase=True,stop_words='english',strip_accents='unicode')
    print(cv.fit_transform(stemmer.stem(db.sms.values[53])))

def features_check(): 
    # check the correlation between features with an heatmap
    corrMatrix = db.corr()
    sns.set( font_scale=1.10 )
    plt.figure( figsize=(10, 10) )
    sns.heatmap( corrMatrix, vmax=.8, linewidths=0.01,
                 square=True, annot=True, cmap='viridis', linecolor="white" )

    plt.title( 'Correlation between features' )
    plt.tight_layout( pad=4 )
    plt.show()

    # check the distribution of word lenght for ham/spam messages
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    g = sns.distplot(a=db[db['label']==0].words)
    p = plt.title('Distribution of words for Ham messages')

    plt.subplot(1, 2, 2)
    g = sns.distplot(a=db[db['label']==1].words, color='red')
    p = plt.title('Distribution of words for Spam messages')

    plt.tight_layout()
    plt.show()

    # check the distribution of currency symbols for ham/spam messages

    plt.figure(figsize=(8,8))
    g = sns.countplot(x='currency', data=db, hue='label')
    p = plt.title('Countplot for currency symbols')
    p = plt.xlabel('0:No, 1:Yes')
    p = plt.ylabel('Count')
    p = plt.legend(labels=['Ham', 'Spam'], loc=9)
    plt.show()

    plt.figure(figsize=(8,8))
    g = sns.countplot(x='numbers', data=db, hue='label')
    p = plt.title('Countplot for numbers')
    p = plt.xlabel('0:No, 1:Yes')
    p = plt.ylabel('Count')
    p = plt.legend(labels=['Ham', 'Spam'], loc=9)
    plt.show()


# plotting with histograms which is the distribution of spam/ham messages in the orginal dataset, training test, and testing set
# this plot gives us a good visualization of the different dataset distribution (the rateo is preserved by stratificated splitting)
def data_distribution():

        figure, axis = plt.subplots(1, 3)
        
        axis[0].set_facecolor('silver')
        axis[0].hist(Y, bins=[0,0.5,1], color='b', rwidth=0.7)
        axis[0].set_xlabel("target value")
        axis[0].set_ylabel("number of instances")
        axis[0].set_title("Dataset distribution")
        axis[0].set_xticks(np.arange(0, 2, 1))
        
        axis[1].set_facecolor('silver')
        axis[1].hist(y_train, bins=[0,0.5,1], color ='g', rwidth=0.7)
        axis[1].set_xlabel("target value")
        axis[1].set_ylabel("number of instances")
        axis[1].set_title("Training set distribution")
        axis[1].set_xticks(np.arange(0, 2, 1))

        axis[2].set_facecolor('silver')
        axis[2].hist(y_test, bins=[0,0.5,1], color= 'r', rwidth=0.7)
        axis[2].set_xlabel("target value")
        axis[2].set_ylabel("number of instances")
        axis[2].set_title("Testing set distribution")
        axis[2].set_xticks(np.arange(0, 2, 1))

        plt.show()

# plot a cloudwords with the most frequent words in the spam messages
def cloud_words():
        spam_messages = db[Y==1]['sms']

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
        spam_messages = db[Y==1]['sms']

        words = ""

        for message in spam_messages:
            
            words += message

        stop = set(STOPWORDS)

        wordcloud = WordCloud(width=800, height=400, max_font_size=100,normalize_plurals=True,min_word_length=3, max_words=50,colormap='summer',stopwords=stop).generate(words)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

#return the top 20 words with their frequency 
def top_n_words(corpus, n):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

#return the top 20 words with their frequency (without stopwords) 
def top_n_words_stopwords(corpus, n):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def print_top_words():

    figure, axis = plt.subplots(2)

    common_words = top_n_words(X['sms'], 20)
    df1 = pd.DataFrame(common_words, columns = ['sms_words' , 'count'])
    df1.groupby('sms_words').sum()['count'].sort_values(ascending=False)
    axis[0].bar(x=df1['sms_words'],height=df1['count'])
    axis[0].set_title('Frequency of words in all the sms')


    common_words = top_n_words_stopwords(X['sms'], 20)
    df1 = pd.DataFrame(common_words, columns = ['sms_words' , 'count'])
    df1.groupby('sms_words').sum()['count'].sort_values(ascending=False)
    axis[1].bar(x=df1['sms_words'],height=df1['count'])
    axis[1].set_title('Frequency of words in all the sms (without stopwords)')

    plt.show()

    figure, axis = plt.subplots(2)

    spam_messages = db[Y==1]['sms']

    common_words = top_n_words(spam_messages, 20)
    df1 = pd.DataFrame(common_words, columns = ['sms_words' , 'count'])
    df1.groupby('sms_words').sum()['count'].sort_values(ascending=False)
    axis[0].bar(x=df1['sms_words'],height=df1['count'],color = 'red')
    axis[0].set_title('Frequency of words in the spam messages')


    common_words = top_n_words_stopwords(spam_messages, 20)
    df1 = pd.DataFrame(common_words, columns = ['sms_words' , 'count'])
    df1.groupby('sms_words').sum()['count'].sort_values(ascending=False)
    axis[1].bar(x=df1['sms_words'],height=df1['count'],color = 'red')
    axis[1].set_title('Frequency of words in the spam messages (without stopwords)')

    plt.show()


if __name__ == '__main__':
    features_check()
    text_conversion_test()
    data_distribution()
    cloud_words()
    cloud_words_stemmed()
    print_top_words()