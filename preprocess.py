import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
from scipy import sparse

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


    # print("X.shape[0]")
    # print(X.shape[0])

    # " take only the first 1000 rows for testing purposes "
    # num_rows = 10000
    # X = X.head(num_rows)
    # y = y.head(num_rows)


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




    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Making X usable by making it numerical.
    # It becomes a scipy sparse matrix.
    # A row has zeros everywhere, except for the columns representing the words in it's word list.
    vectorizer = TfidfVectorizer()
    X_train_TF_IDF = vectorizer.fit_transform(X_train['HeadlineAndDesc'])
    X_test_TF_IDF = vectorizer.transform(X_test['HeadlineAndDesc'])

    # print(X_train_TF_IDF)

    # print the shape of X_train_TF_IDF
    print("X_train_TF_IDF.shape")
    print(X_train_TF_IDF.shape)




    """# get quartiles of nonzero data in X_train_TF_IDF
    # make sure it only accounts for nonzero data
    
    # print max of X_train_TF_IDF
    print(np.max(X_train_TF_IDF.data))

    # It seems np.percentile already ignores all the zeros:
    print(np.percentile(X_train_TF_IDF.data, 1))
    print(np.percentile(X_train_TF_IDF.data, 90))
    print(np.percentile(X_train_TF_IDF.data, 100))

    B = X_train_TF_IDF > np.percentile(X_train_TF_IDF.data, 90)
    print("B")
    print(B)
    ixs = sp.sparse.find(B)

    # You could do this to retain original data:
    a = X_train_TF_IDF_discretised.data
    X_train_TF_IDF_discretised.data = np.where(a > np.percentile(X_train_TF_IDF.data, 25), a, 0)
    """
    
    # sp.sparse.csr_matrix(X_train_TF_IDF.shape, dtype=np.int32)
    X_train_TF_IDF_discretised = X_train_TF_IDF.copy()
    X_train_TF_IDF_discretised.data[:] = 0
    X_train_TF_IDF_discretised.data = X_train_TF_IDF_discretised.data.astype(np.int32)
    print("np.unique(X_train_TF_IDF_discretised.data)")
    print(np.unique(X_train_TF_IDF_discretised.data))






    X_TF_data = X_train_TF_IDF.data

    nextPercent = np.percentile(X_train_TF_IDF.data, 25)
    a = np.where((nextPercent > X_TF_data), 1, 0)
    # print("X_train_TF_IDF_discretised.data")
    # print(X_train_TF_IDF_discretised.data)
    # print("a")
    # print(a)
    X_train_TF_IDF_discretised.data = X_train_TF_IDF_discretised.data + a

    percentiles = [25, 50, 75]
    for ix in range(len(percentiles)-1):
        lowerPercent = np.percentile(X_TF_data, percentiles[ix])
        nextPercent = np.percentile(X_TF_data, percentiles[ix+1])
        a = np.where((nextPercent > X_TF_data) & (X_TF_data > lowerPercent), ix+2, 0)
        X_train_TF_IDF_discretised.data += a


    lowerPercent = np.percentile(X_train_TF_IDF.data, 75)
    a = np.where((X_TF_data > lowerPercent), 4, 0)
    X_train_TF_IDF_discretised.data += a


    print("np.unique(X_train_TF_IDF_discretised.data)")
    print(np.unique(X_train_TF_IDF_discretised.data))
    
    
    print("X_train_TF_IDF_discretised")
    print(X_train_TF_IDF_discretised)

    print("X_train_TF_IDF_discretised.data")
    print(X_train_TF_IDF_discretised.data)





    input("Stop A")

    if False:
        
        sparse.save_npz("X_train_TF_IDF_discretised.npz", X_train_TF_IDF_discretised)

        # second_X_train_TF_IDF_discretised = sparse.load_npz("X_train_TF_IDF_discretised.npz")
        # print("second_X_train_TF_IDF_discretised")
        # print(second_X_train_TF_IDF_discretised)

        input("Stop B")



        from sklearn.feature_selection import mutual_info_classif
        mutual_infos = mutual_info_classif(X_train_TF_IDF_discretised, y_train, random_state=42)

        mutual_infos_df = pd.DataFrame(mutual_infos)
        mutual_infos_df.to_csv("mutual_infos_42.csv")
        print(mutual_infos)

        # from sklearn.feature_selection import mutual_info_regression
        # mutual_info_regression(X_train_TF_IDF, y_train)


    input("Stop C")


    # get "mutual_infos_42.csv" from csv file
    mutual_infos_df = pd.read_csv("mutual_infos_42.csv")
    print("mutual_infos_df")
    print(mutual_infos_df)
    mutual_infos = mutual_infos_df.values[:,1].reshape((-1))
    print("mutual_infos")
    print(mutual_infos)

    # print max of mutual_infos
    print("np.max(mutual_infos)")
    print(np.max(mutual_infos)) 



    m_i_sorter = [(mutual_infos[i],i) for i in range(mutual_infos.size)]
    print("m_i_sorter[:10]")
    print(m_i_sorter[:10])
    m_i_sorter = sorted(m_i_sorter, key=lambda x: x[0], reverse=True)
    m_i_sorted = [x[0] for x in m_i_sorter]
    sort_permutation = [x[1] for x in m_i_sorter]

    print("m_i_sorted[:10]")
    print(m_i_sorted[:10])
    print("sort_permutation[:10]")
    print(sort_permutation[:10])

    # print shape of X_train_TF_IDF
    print("X_train_TF_IDF.shape")
    print(X_train_TF_IDF.shape)

    sort_index = np.array(list(sort_permutation))

    best_index = sort_index[:300]

    input("Stop D")

    print("X_train_TF_IDF.shape")
    print(X_train_TF_IDF.shape)
    X_train_TF_IDF_trimmed = X_train_TF_IDF[:,best_index]
    print("X_train_TF_IDF_trimmed")
    print(X_train_TF_IDF_trimmed)
    print("X_train_TF_IDF_trimmed.shape")
    print(X_train_TF_IDF_trimmed.shape)


    input("Stop E")


    sparse.save_npz("X_train_TF_IDF_trimmed.npz", X_train_TF_IDF_trimmed)

    X_train_TF_IDF_trimmed_loaded = sparse.load_npz("X_train_TF_IDF_trimmed.npz")
    print("X_train_TF_IDF_trimmed_loaded")
    print(X_train_TF_IDF_trimmed_loaded)
    print("X_train_TF_IDF_trimmed_loaded.shape")
    print(X_train_TF_IDF_trimmed_loaded.shape)






main()