import pandas as pd
import chardet
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# This function load the dataset, converts the categorical labels, splits (with stratification) training and testing set and return the two subsets
# this function will be called by all the other classes that perform classification 
def get_data():

    # check which is the actual encoding of the .csv file 
    rawdata = open('dataset/spam.csv', "rb").read()
    print(chardet.detect(rawdata))

    # reading the csv file into a pandas dataframe using the right encoding
    db = pd.read_csv(r'dataset/spam.csv',encoding='Windows-1252')

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

    # labels and features raw arrays
    Y = db['label']
    X = db['sms']
    
    # split testing and training set (70/30) with stratification on the label
    X_train, X_test, y_train, y_test = train_test_split(X,Y,stratify = Y, test_size = 0.3)

    # return the labels and features array (X and Y) and the splitted version of them (train 70%, test 30%)
    return db,X,Y,X_train,X_test,y_train,y_test