import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

printout = False


df = pd.read_csv("dataFrameOfImportants.csv")

print(df.shape)
df = df.dropna()
print(df.shape)


X = df[['HeadlineAndDesc']]
y = df['Category']





# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Making X usable by making it numerical.
# It becomes a scipy sparse matrix.
# A row has zeros everywhere, except for the columns representing the words in it's word list.
vectorizer = TfidfVectorizer()
X_train_TF_IDF = vectorizer.fit_transform(X_train['HeadlineAndDesc'])
X_test_TF_IDF = vectorizer.transform(X_test['HeadlineAndDesc'])


if printout:
    
    # print(X_train_TF_IDF)

    # print the shape of X_train_TF_IDF
    print("X_train_TF_IDF.shape")
    print(X_train_TF_IDF.shape)





# get "mutual_infos_42.csv" from csv file
mutual_infos_df = pd.read_csv("mutual_infos_42.csv")
# print("mutual_infos_df")
# print(mutual_infos_df)
mutual_infos = mutual_infos_df.values[:,1].reshape((-1))

if printout:
    print("mutual_infos")
    print(mutual_infos)



m_i_sorter = [(mutual_infos[i],i) for i in range(mutual_infos.size)]
m_i_sorter = sorted(m_i_sorter, key=lambda x: x[0], reverse=True)
m_i_sorted = [x[0] for x in m_i_sorter]
sort_permutation = [x[1] for x in m_i_sorter]

sort_index = np.array(list(sort_permutation))

best_index = sort_index[:300]

X_train_TF_IDF_trimmed = X_train_TF_IDF[:,best_index]
X_test_TF_IDF_trimmed = X_test_TF_IDF[:,best_index]

if printout:
    print("X_train_TF_IDF_trimmed.shape")
    print(X_train_TF_IDF_trimmed.shape)
    print("X_test_TF_IDF_trimmed.shape")
    print(X_test_TF_IDF_trimmed.shape)





import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


"""
# http://pytorch.org/docs/master/sparse.html#construction-of-csr-tensors
print("X_train_TF_IDF_trimmed.indices")
print(X_train_TF_IDF_trimmed.indices)
print("X_train_TF_IDF_trimmed.indptr")
print(X_train_TF_IDF_trimmed.indptr)"""


X_train_coo = X_train_TF_IDF_trimmed.tocoo()
X_test_coo = X_test_TF_IDF_trimmed.tocoo()


"""
# https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor
values = coo.data
indices = np.vstack((coo.row, coo.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = coo.shape
torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
"""


i = np.vstack((X_train_coo.row, X_train_coo.col))
v = X_train_coo.data
s = torch.sparse_coo_tensor(i, v, X_train_coo.shape)

X_train_TF_IDF_trimmed_10th_diagonal = X_train_TF_IDF_trimmed[10,10]
print("X_train_TF_IDF_trimmed_10th_diagonal")
print(X_train_TF_IDF_trimmed_10th_diagonal)

s_10th_diagonal = s[10,10]
print("s_10th_diagonal")
print(s_10th_diagonal)




square_start = 10
square_stop = 15


X_train_TF_IDF_trimmed_upper_left_corner = X_train_TF_IDF_trimmed[square_start:square_stop,square_start:square_stop].todense()
print("X_train_TF_IDF_trimmed_upper_left_corner")
print(X_train_TF_IDF_trimmed_upper_left_corner)


# # doesn't work. Slicing isn't possible with (?sparse?) tensors.
# s_trimmed = s[:,best_index]
# print("s_trimmed")
# print(s_trimmed)

def tens_slice(tens, zeroth_indices, first_indices):
    zeroth_indices = torch.LongTensor(zeroth_indices)
    first_indices = torch.LongTensor(first_indices)
    ret_tens = torch.index_select(tens, 0, zeroth_indices)
    ret_tens = torch.index_select(ret_tens, 1, first_indices)
    return ret_tens

square_ixs = range(square_start, square_stop)
s_upper_left_corner = tens_slice(s, square_ixs, square_ixs).to_dense()
print("s_upper_left_corner")
print(s_upper_left_corner)

# print("X_train_TF_IDF_trimmed.max()")
# print(X_train_TF_IDF_trimmed.max())
# # Ne dela:
# print("torch.max(s)")
# print(torch.max(s))

"""
# https://discuss.pytorch.org/t/sparsemax-in-pytorch/4968
# from sparsemax import Sparsemax
from sparsemax import Sparsemax
sparsemax = Sparsemax(dim=-1)"""

"""
sparse.sum  	

Return the sum of each row of the given sparse tensor.
https://pytorch.org/docs/stable/sparse.html"""



print(s)