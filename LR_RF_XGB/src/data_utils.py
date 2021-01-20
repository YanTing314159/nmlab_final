import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def preprocess_dropuk(data_dir, norm=True, return_uk=False):
  #====load data
  print("loading data...")
  df1 = pd.read_csv(data_dir + "elliptic_txs_features.csv", header=None)
  data = df1.to_numpy()
  #print(data.shape) #(203769, 167)
  df2 = pd.read_csv(data_dir + "elliptic_txs_classes.csv", header=None)
  label = df2[:].to_numpy()
  label = label[1:]#first row is [txId, class]
  #print(label.shape) #(203769, 2)
  clean_data = []
  clean_label = []
  label_count = {}
  uk_data = []
  for i in range(0, label.shape[0]):
    if data[i, 0] != int(label[i, 0]):
      print("WARING! inequal txid")
    if label[i][1] == 'unknown':
      uk_data.append(data[i])
    else:
      clean_data.append(data[i])
      clean_label.append(label[i])
      if label[i][1] not in label_count:
        label_count[label[i][1]] = 0
      else:
        label_count[label[i][1]] += 1
  #print(label_count) #{'2': 42018, '1': 4544}
  clean_data = np.array(clean_data)
  clean_label = np.array(clean_label)
  #print(clean_data.shape) #(46564, 167)
  #print(clean_label.shape) #(46564, 2)
  norm_data = (clean_data - clean_data.mean(0)) / clean_data.std(0)
  #print(norm_data.shape) #(46564, 167)
  print("data loaded")
  if norm and not return_uk:
    return norm_data, clean_label
  elif not norm and not return_uk:
    return clean_data, clean_label
  elif norm and return_uk:
    uk_data = np.array(uk_data)
    uk_data = (uk_data - uk_data.mean(0)) / uk_data.std(0)
    return norm_data, clean_label, uk_data
  else:
    uk_data = np.array(uk_data)
    return clean_data, clean_label, uk_data
  

def preprocess_dropuk_split(data, label):
  clean_data_1 = []
  clean_label_1 = []
  clean_data_2 = []
  clean_label_2 = []
  for i in range(data.shape[0]):
    if int(label[i][1]) == 1:
      clean_data_1.append(data[i])
      clean_label_1.append(label[i])
    elif int(label[i][1]) == 2:
      clean_data_2.append(data[i])
      clean_label_2.append(label[i])
    else:
      print("unexpect label! in preprocess_dropuk_split:", label[i][1])
  clean_data_1 = np.array(clean_data_1)
  clean_label_1 = np.array(clean_label_1)
  clean_data_2 = np.array(clean_data_2)
  clean_label_2 = np.array(clean_label_2)

  return clean_data_1, clean_label_1, clean_data_2, clean_label_2

def infinite_iter(iterable):
  it = iter(iterable)
  while True:
    try:
      ret = next(it)
      yield ret
    except StopIteration:
      it = iter(iterable)

class myDataset(Dataset):
  def __init__(self, data, label):
    self.data = data
    self.label = label

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    x = self.data[idx][2:]
    y = str(self.label[idx, 1]).strip()

    if y == "unknown":
      y = [0]
    else:
      y = [int(y)]
    x = torch.Tensor(x).float()
    y = torch.Tensor(y).long()
    return (x, y) #y:0->unknown, 1->illicit, 2->licit

def get_data_iterater(dataset, batch_size, shuffle, drop_last=True):
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=drop_last)
  return infinite_iter(loader)


