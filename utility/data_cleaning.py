import pandas as pd
import chardet
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk

# This function load the dataset, converts the categorical labels, splits (with stratification) training and testing set and return the two subsets
# this function will be called by all the other classes that perform classification 
def get_data():

    # reading the csv file into a pandas dataframe using the right encoding
    db = pd.read_csv(r'dataset/spam.csv',encoding="ISO-8859-1")

    # convert the categorical labels of v1 (ham,spam) in binary (0,1) labels
    db[['v1']] = db[['v1']].apply(LabelEncoder().fit_transform)

    # dropping the three unused features (they mostly have null values)
    #print(db.isnull().sum())
    db.drop(labels=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

    # renaming the features with meaningful names
    db.columns = ['label','sms']

    # checking the label distribution and dropping duplicates (if there are any)
    print(db.label.value_counts())
    db.drop_duplicates(inplace=True)
    print(db.label.value_counts())

    # function that check if currency symbols are present in a string
    def currency(x):
        currency_symbols = ['€', '$', '¥', '£', '₹']
        for i in currency_symbols:
            if i in x:
                return 1
        return 0

    # function that check if numbers are present in a string
    def numbers(x):
        for i in x:
            if ord(i)>=48 and ord(i)<=57:
                return 1
        return 0

    def links(x):
        link_strings = ['http','https','www.','.com','.uk']
        for i in link_strings:
            if x.find(i) != -1:
                return 1
        return 0

    # adding some meaningful features to the data: number of chars, number of words, presence of numbers, links and currency symbols.
    db['chars'] = db['sms'].apply(len)
    db['words'] = db.apply(lambda row: nltk.word_tokenize(row['sms']), axis=1).apply(len)
    db['sentences'] = db.apply(lambda row: nltk.sent_tokenize(row["sms"]), axis=1).apply(len)
    db['currency'] = db['sms'].apply(currency)
    db['numbers']=db['sms'].apply(numbers)
    db['links']=db['sms'].apply(links)

    # labels and features raw arrays
    Y = db['label']
    X = db.drop(['label'],axis=1)
    
    # split testing and training set (70/30) with stratification on the label
    X_train, X_test, y_train, y_test = train_test_split(X,Y,stratify = Y, test_size = 0.3)

    # return the labels and features array (X and Y) and the splitted version of them (train 70%, test 30%)
    return db,X,Y,X_train,X_test,y_train,y_test