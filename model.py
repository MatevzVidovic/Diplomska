import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report



def main():

    df = pd.read_csv("dataFrameOfImportants.csv")

    # print(df)

    #I don't know how na values are possible despite the preprocessing,
    # but somehow they exist. And we need to do these steps for the algorithms to work.
    print(df.shape)
    df = df.dropna()
    print(df.shape)


    X = df[['HeadlineAndDesc']]
    y = df['Category']



    # retain only firts 100 columns of X
    X = X.iloc[:, :10]

    
    # This would make y into numbers:
    
    
    # get all distinct values of y
    print(y.unique())

    # make a dictionary between unique values and numbers, and transform y so it is a vector of corresponding numbers
    y_dict = {}
    for i, category in enumerate(y.unique()):
        y_dict[category] = i
    
    y = y.apply(lambda x: y_dict[x])

    print(y.unique())
    print(y)




    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    # from sklearn.feature_selection import mutual_info_classif
    # mutual_infos = mutual_info_classif(X_train, y_train)

    # print(mutual_infos)



    """
        todo: trim columns from x which aren't good enough
    """




    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Making X usable by making it numerical.
    # It becomes a scipy sparse matrix.
    # A row has zeros everywhere, except for the columns representing the words in it's word list.
    vectorizer = TfidfVectorizer()
    X_train_TF_IDF = vectorizer.fit_transform(X_train['HeadlineAndDesc'])
    X_test_TF_IDF = vectorizer.transform(X_test['HeadlineAndDesc'])

    # print(X_train_TF_IDF)


    from sklearn.feature_selection import mutual_info_regression
    mutual_info_regression(X_train_TF_IDF, y_train)



main()