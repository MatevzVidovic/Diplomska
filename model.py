import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from timeit import default_timer as timer





printout = False

epochs = 1


test_purposes = False

test_num_of_train_rows = 10000
test_num_of_test_rows = 10000



types_dict = {
    0 : "basic",
    1 : "deep_3pure_middle",
    2 : "deep_4pure_middle_dropout_leaky_relu",
}

model_type = types_dict[2]
chosen_num_of_features = 5000
second_layer_size = 1000
middle_layer_size = 700

dropout = 0.1
leaky_relu_alpha = 0.1

"""
!!!!!!!!!!!!!!!!!!!
Res upam da je pri
nn.Softmax(dim=1)
dim pravilno izbran.

Pomoje ne, ker terrible rezultati.

Zdaj dajem na 0.
No, tudi tako dobim zelo zelo slabe rezultate.

Odstranil sem softmax.
!!!!!!!!!!!!!!!!!!!"""

# nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()

# optimizer pa moras urejati spodaj




model_data_path = model_type + "_" + str(chosen_num_of_features) + "_" + str(second_layer_size) + "_" + str(middle_layer_size) + "_model/"


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# load_previous_model has been moved down, so that it works with while True










# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        """
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        PREJ SEM IMEL NA ZADNJEM
        nn.Linear(300, 14)

        DOBIL SEM PODOBNO NAPAKO KOT SPODAJ. PO NASVETU SEM POGNAL KOT
        CUDA_LAUNCH_BLOCKING=1 python3 model.py
        PA JE BILA ISTA NAPAKA.

        KOT KAZE SEM MISMATCHAL STEVILO TARGET VARIABLEOV

        https://discuss.pytorch.org/t/runtimeerror-cuda-error-device-side-assert-triggered/34213/9
        
        https://discuss.pytorch.org/t/how-to-fix-cuda-error-device-side-assert-triggered-error/137553


        Epoch 1
-------------------------------
../aten/src/ATen/native/cuda/Loss.cu:250: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [5,0,0] Assertion `t >= 0 && t < n_classes` failed.
Traceback (most recent call last):
  File "/home/matevzvidovic/Desktop/SeminarskaDemo/model.py", line 492, in <module>
    train(train_dataloader, model, loss_fn, optimizer)
  File "/home/matevzvidovic/Desktop/SeminarskaDemo/model.py", line 440, in train
    loss = loss_fn(pred, y)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/loss.py", line 1179, in forward
    return F.cross_entropy(input, target, weight=self.weight,
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/functional.py", line 3059, in cross_entropy
    return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
RuntimeError: CUDA error: device-side assert triggered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.



        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """




        """
        
Sparse Layers

nn.Embedding
	

A simple lookup table that stores embeddings of a fixed dictionary and size.

nn.EmbeddingBag
	

Compute sums or means of 'bags' of embeddings, without instantiating the intermediate embeddings.
Distance Functions

nn.CosineSimilarity
	

print(f"Using {device} device")
Returns cosine similarity between x1x1 and x2x2 computed along dim."""
        
        
        if model_type == "basic":
            self.sequential_NN_stack = nn.Sequential(
                nn.Linear(chosen_num_of_features, second_layer_size),
                nn.ReLU(),
                nn.Linear(second_layer_size, middle_layer_size),
                nn.ReLU(),
                nn.Linear(middle_layer_size, len(categories))
            )
        elif model_type == "deep_3pure_middle":
            self.sequential_NN_stack = nn.Sequential(
              nn.Linear(chosen_num_of_features, second_layer_size),
              nn.ReLU(),
              nn.Linear(second_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Linear(middle_layer_size, len(categories))
          )
        elif model_type == "deep_4pure_middle_dropout_leaky_relu":
            self.sequential_NN_stack = nn.Sequential(
              nn.Linear(chosen_num_of_features, second_layer_size),
              nn.ReLU(),
              nn.Linear(second_layer_size, middle_layer_size),
              nn.LeakyReLU(leaky_relu_alpha),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Dropout(p=dropout),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.LeakyReLU(leaky_relu_alpha),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.LeakyReLU(leaky_relu_alpha),
              nn.Linear(middle_layer_size, middle_layer_size),
              nn.ReLU(),
              nn.Linear(middle_layer_size, len(categories)),
              # nn.Softmax(dim=0)
          )
            
            

    def forward(self, x):
        x = self.flatten(x)
        logits = self.sequential_NN_stack(x)

        # m = nn.Softmax(dim=1)
        # print("Softmax(dim=1)")
        # print(m(logits))

        # m = nn.Softmax(dim=0)
        # print("Softmax(dim=0)")
        # print(m(logits))

        return logits
































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
categories = []
for i, category in enumerate(y.unique()):
    category2hash[category] = i
    categories.append(category)

y = y.apply(lambda x: category2hash[x])

if printout:
    print(y.unique())
    print(y)



"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
random_state mora ostati 42 !!!!!!!!!!!!!
Sicer:
File "/home/matevzvidovic/Desktop/SeminarskaDemo/model.py", line 141, in <module>
    X_train_TF_IDF_trimmed = X_train_TF_IDF[:,best_index]
IndexError: index (56545) out of range

ker je drugačno število besed v tem naboru in se lahko to zgodi.

To sedaj zal ne deluje vec:

# if it is none, than (test_purposes = True) doesn't always take the same examples, which is nice
train_test_split_seed = None # 42

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Making X usable by making it numerical.
# It becomes a scipy sparse matrix.
# A row has zeros everywhere, except for the columns representing the words in it's word list.
vectorizer = TfidfVectorizer()
X_train_TF_IDF = vectorizer.fit_transform(X_train['HeadlineAndDesc'])
X_test_TF_IDF = vectorizer.transform(X_test['HeadlineAndDesc'])

if test_purposes:
    X_train_TF_IDF = X_train_TF_IDF[:test_num_of_train_rows, :]
    y_train = y_train[:test_num_of_train_rows]
    X_test_TF_IDF = X_test_TF_IDF[:test_num_of_test_rows, :]
    y_test = y_test[:test_num_of_test_rows]

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

best_index = sort_index[:chosen_num_of_features]

X_train_TF_IDF_trimmed = X_train_TF_IDF[:,best_index]
X_test_TF_IDF_trimmed = X_test_TF_IDF[:,best_index]

if printout:
    print("X_train_TF_IDF_trimmed.shape")
    print(X_train_TF_IDF_trimmed.shape)
    print("X_test_TF_IDF_trimmed.shape")
    print(X_test_TF_IDF_trimmed.shape)












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




"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

TU MORA BIT
dtype=torch.float32
KER PYTORCH V NN.LINEAR UPORABLJA FLOAT32, V OSNOVI SO PA TO FLOAT64 (DOUBLE)

https://discuss.pytorch.org/t/runtimeerror-mat1-and-mat2-must-have-the-same-dtype/166759/7

IN DOBIM TA ERROR:
Epoch 1
-------------------------------
Traceback (most recent call last):
  File "/home/matevzvidovic/Desktop/SeminarskaDemo/model.py", line 420, in <module>
    train(train_dataloader, model, loss_fn, optimizer)
  File "/home/matevzvidovic/Desktop/SeminarskaDemo/model.py", line 367, in train
    pred = model(X)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/matevzvidovic/Desktop/SeminarskaDemo/model.py", line 333, in forward
    logits = self.linear_relu_stack(x)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

i = np.vstack((X_train_coo.row, X_train_coo.col))
v = X_train_coo.data
X_train_tens = torch.sparse_coo_tensor(i, v, X_train_coo.shape, dtype=torch.float32)

i = np.vstack((X_test_coo.row, X_test_coo.col))
v = X_test_coo.data
X_test_tens = torch.sparse_coo_tensor(i, v, X_test_coo.shape, dtype=torch.float32)

y_train_tens = torch.tensor(y_train.values.astype("int8")).type(torch.LongTensor)
y_test_tens = torch.tensor(y_test.values.astype("int8")).type(torch.LongTensor)



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




# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



model = NeuralNetwork().to(device)
print(model)







while True:




  # True if directory already exists
  load_previous_model = os.path.isdir(model_data_path)
  # Set to False if you want to rewrite data
  load_previous_model = load_previous_model






  if load_previous_model:
    prev_model_details = pd.read_csv(model_data_path + "previous_model_" + str(chosen_num_of_features) + "_details.csv")
    prev_serial_num = prev_model_details["previous_serial_num"][0]
    prev_cumulative_epochs = prev_model_details["previous_cumulative_epochs"][0]
    model.load_state_dict(torch.load(model_data_path + "model_" + str(chosen_num_of_features) + "_" + str(prev_serial_num) + ".pth"))
  else:
    prev_serial_num = 0
    prev_cumulative_epochs = 0
        
      



  # https://pytorch.org/docs/stable/optim.html
  # SGD - stochastic gradient descent
  # imajo tudi Adam, pa sparse adam, pa take.
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)







  # ...

  train_times = []


  def train(dataloader, model, loss_fn, optimizer):
      size = len(dataloader.dataset)
      model.train()
      
      start = timer()

      for batch, (X, y) in enumerate(dataloader):

          """
          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

          !!!!!! DODAJ:
          y = y.type(torch.LongTensor)


          TOLE JE STARA NAPAKA, AMPAK JO BOM PUSTIL, DA JE NE RABIM SE ZGORAJ PISAT, KER BI BILO BREZVEZE.

          SEDAJ NAREDIM TO KONERZIJO ZE ZGORAJ, KER SEM PO TRENIRANJU DOBIL
          RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Char'
          IN JE BILO KER JE Z Y_TRAIN ENAK PROBLEM

          
          https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for

          ker
          Epoch 1
  -------------------------------
  Traceback (most recent call last):
    File "/home/matevzvidovic/Desktop/SeminarskaDemo/model.py", line 461, in <module>
      train(train_dataloader, model, loss_fn, optimizer)
    File "/home/matevzvidovic/Desktop/SeminarskaDemo/model.py", line 409, in train
      loss = loss_fn(pred, y)
    File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
      return self._call_impl(*args, **kwargs)
    File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
      return forward_call(*args, **kwargs)
    File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/modules/loss.py", line 1179, in forward
      return F.cross_entropy(input, target, weight=self.weight,
    File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/nn/functional.py", line 3059, in cross_entropy
      return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
  RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Char'


          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          """
          X, y = X.to(device), y.to(device)

          # Compute prediction error
          pred = model(X)
          loss = loss_fn(pred, y)

          # Backpropagation
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          if batch % 100 == 0:
              end = timer()
              train_times.append(end - start)
              start = timer()
              loss, current = loss.item(), (batch + 1) * len(X)
              print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")











  accuracies = []
  avg_losses = []
  def test(dataloader, model, loss_fn):
      size = len(dataloader.dataset)
      num_batches = len(dataloader)
      model.eval()
      test_loss, correct = 0, 0
      with torch.no_grad():
          for X, y in dataloader:
              X, y = X.to(device), y.to(device)
              pred = model(X)
              test_loss += loss_fn(pred, y).item()
              correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      test_loss /= num_batches
      correct /= size
      print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
      accuracies.append("{(100*correct):>0.1f}%")
      avg_losses.append("{test_loss:>8f}")










  all_train_times = []

  for t in range(epochs):
      print(f"Epoch {t+1}\n-------------------------------")
      train(train_dataloader, model, loss_fn, optimizer)
      del train_times[0]
      print(train_times)
      all_train_times.extend(train_times)
      train_times = []
      test(test_dataloader, model, loss_fn)
  print("Done!")





  try:
    os.mkdir(model_data_path)
  except:
    pass

  torch.save(model.state_dict(), model_data_path + "model_" + str(chosen_num_of_features) + "_" + str(prev_serial_num+1) + ".pth")

  new_df = pd.DataFrame({"previous_serial_num": [prev_serial_num+1], "previous_cumulative_epochs": [prev_cumulative_epochs+epochs]})
  new_df.to_csv(model_data_path + "previous_model_" + str(chosen_num_of_features) + "_details.csv")

  """
  Razlog za spodnji nacin kode:
  File "/home/matevzvidovic/.local/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 677, in _extract_index
    raise ValueError("All arrays must be of the same length")
ValueError: All arrays must be of the same length
"""
  a = {"train_times": all_train_times, "accuracies": accuracies, "avg_losses": avg_losses}
  new_df = pd.DataFrame.from_dict(a, orient='index')
  new_df = df.transpose()
  new_df.to_csv(model_data_path + "train_times_" + str(prev_serial_num+1) + ".csv")

  print("Saved PyTorch Model State")













  """model = NeuralNetwork().to(device)
  model.load_state_dict(torch.load("model.pth"))

  classes = categories

  model.eval()
  x, y = test_data[0][0], test_data[0][1]
  with torch.no_grad():
      x = x.to(device)
      pred = model(x)
      predicted, actual = classes[pred[0].argmax(0)], classes[y]
      print(f'Predicted: "{predicted}", Actual: "{actual}"')"""