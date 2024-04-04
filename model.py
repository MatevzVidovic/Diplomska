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


if printout:
    # get all distinct values of y
    print(y.unique())

# make a dictionary between unique values and numbers, and transform y so it is a vector of corresponding numbers
category2hash = {}
for i, category in enumerate(y.unique()):
    category2hash[category] = i

y = y.apply(lambda x: category2hash[x])

if printout:
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

from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def tens_slice(tens, zeroth_indices=None, first_indices=None):
    if zeroth_indices is None:
        zeroth_indices = range(tens.size(dim=0))
    if first_indices is None:
        first_indices = range(tens.size(dim=1))

    zeroth_indices = torch.LongTensor(zeroth_indices)
    first_indices = torch.LongTensor(first_indices)
    ret_tens = torch.index_select(tens, 0, zeroth_indices)
    ret_tens = torch.index_select(ret_tens, 1, first_indices)
    return ret_tens






"""
# http://pytorch.org/docs/master/sparse.html#construction-of-csr-tensors
print("X_train_TF_IDF_trimmed.indices")
print(X_train_TF_IDF_trimmed.indices)
print("X_train_TF_IDF_trimmed.indptr")
print(X_train_TF_IDF_trimmed.indptr)"""


X_train_coo = X_train_TF_IDF_trimmed.tocoo()
X_test_coo = X_test_TF_IDF_trimmed.tocoo()




i = np.vstack((X_train_coo.row, X_train_coo.col))
v = X_train_coo.data
X_train_tens = torch.sparse_coo_tensor(i, v, X_train_coo.shape)

i = np.vstack((X_test_coo.row, X_test_coo.col))
v = X_test_coo.data
X_test_tens = torch.sparse_coo_tensor(i, v, X_test_coo.shape)

y_train_tens = torch.tensor(y_train.values.astype("int8"))
y_test_tens = torch.tensor(y_test.values.astype("int8"))



if False:


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









class CustomDataset(Dataset):
    def __init__(self, X_TF_IDF_sparse_tensor, target_var_tensor, transform=None, target_transform=None):
        self.data = X_TF_IDF_sparse_tensor
        self.target_var = target_var_tensor
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.target_var)

    def __getitem__(self, idx):


        """
        !!!!!!!!!!!!!!!!!!!!!!!!!!
        TU MORA BIT .to_dense()

        sicer dobim
        File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/utils/data/_utils/collate.py", line 164, in collate_tensor_fn
        raise RuntimeError(
        RuntimeError: Batches of sparse tensors are not currently supported by the default collate_fn; please provide a custom collate_fn to handle them appropriately.

        In bi bilo treba v DataLoader dati collate_fn=custom_collate
        in ga napisati pravilno.

        !!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        data_row = tens_slice(self.data, range(idx,idx+1), None).to_dense()
        target = self.target_var[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return data_row, target



train_data = CustomDataset(X_train_tens, y_train_tens)
test_data = CustomDataset(X_test_tens, y_test_tens)



batch_size = 64


train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break