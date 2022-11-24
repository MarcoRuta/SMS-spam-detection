import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# reading the csv file into a pandas dataframe
db = pd.read_csv (r'spam.csv',encoding='latin-1')

# convert the categorical labels of v1 (ham,spam) in binary (0,1) labels
db[['v1']] = db[['v1']].apply(LabelEncoder().fit_transform)

# split testing and training set (70/30) with stratification on the label
X_train, X_test, y_train, y_test = train_test_split(db['v2'],db['v1'],stratify = db['v1'], test_size = 0.3)


# plotting with histograms which is the distribution of spam/ham messages in the orginal dataset, training test, and testing set
# this plot gives us a good visualization of the different dataset distribution (the rateo is preserved by stratificated splitting)
figure, axis = plt.subplots(1, 3)

axis[0].hist(db['v1'], bins=[0,0.5,1], color='b', rwidth=0.7)
axis[0].set_xlabel("target value")
axis[0].set_ylabel("number of instances")
axis[0].set_title("Dataset distribution")

axis[1].hist(y_train, bins=[0,0.5,1], color ='g', rwidth=0.7)
axis[1].set_xlabel("target value")
axis[1].set_ylabel("number of instances")
axis[1].set_title("Training set distribution")

axis[2].hist(y_test, bins=[0,0.5,1], color= 'r', rwidth=0.7)
axis[2].set_xlabel("target value")
axis[2].set_ylabel("number of instances")
axis[2].set_title("Testing set distribution")

plt.show()