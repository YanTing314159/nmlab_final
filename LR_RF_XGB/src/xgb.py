import numpy as np
import argparse
from data_utils import preprocess_dropuk
from utils import get_score
import xgboost as xgb
#for pseudo labeling
import pandas as pd

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-data_dir', default='./data/')

  args = parser.parse_args()
  #====load data
  clean_data, clean_label, uk_data = preprocess_dropuk(args.data_dir, norm=False, return_uk=True)
  concate_data = np.c_[clean_data.reshape(len(clean_data), -1), clean_label[:,1].reshape(len(clean_label), -1)]
  """
  random.seed(0)
  np.random.seed(2)
  shuffle_data = np.random.permutation(concate_data)
  """
  param = {}
  param['booster'] = 'gbtree'
  param['objective'] = 'binary:logistic'
  param['eta'] = 0.7
  param['gamma'] = 0
  param['max_depth'] = 10
  param['min_child_weight']=1
  param['max_delta_step'] = 0
  param['subsample']= 1
  param['colsample_bytree']=1
  param['silent'] = 1
  param['seed'] = 0
  param['base_score'] = 0.5

  xgbc = xgb.XGBClassifier()
  xgbc.set_params(**param)
  #dtrain = xgb.DMatrix(concate_data[:30000, :167], label=concate_data[:30000, -1])
  validation_split_rate = 0.0
  validation_size = int(concate_data.shape[0]*validation_split_rate)
  test_split_rate = 0.3
  test_size = int(concate_data.shape[0]*test_split_rate)
  print("building xgboost...")
  xgbc.fit(concate_data[:-(validation_size+test_size), 2:94], concate_data[:-(validation_size+test_size), -1])
  print("train set")
  pred_labels = xgbc.predict(concate_data[:-(validation_size+test_size), 2:94]) #train
  get_score(pred_labels, concate_data[:-(validation_size+test_size), -2:])
  """
  print("validation set")
  pred_labels = xgbc.predict(concate_data[-(validation_size+test_size):-test_size, 2:167]) #train
  get_score(pred_labels, concate_data[-(validation_size+test_size):-test_size, -2:])
  """
  print("test set")
  pred_labels = xgbc.predict(concate_data[-test_size:, 2:94]) #test
  get_score(pred_labels, concate_data[-test_size:, -2:])

  #pred for single sample
  #pred_labels = xgbc.predict(concate_data[0, :165].reshape(-1, 167))
  #print(pred_labels) #['2'] or ['1']

  pseudo_labeling=True
  if pseudo_labeling:

    pred_labels = xgbc.predict(uk_data[:, 2:167])
    #print(pred_labels.shape) #(157205,)
    #print(uk_data.shape) #(157205, 167)
    concate_uk_data = np.c_[uk_data.reshape(len(uk_data), -1), pred_labels.reshape(len(pred_labels), -1)].astype(int)
    #print(concate_uk_data.shape) #(157205, 168)
    concate_data = np.concatenate((concate_uk_data, concate_data), axis=0)
    #print(concate_data.shape)#(157205, 168) + (46564, 168) = (203769, 168)
    print("building random forest with pseudo labeling...")
    xgbc_pl = xgb.XGBClassifier()
    xgbc_pl.set_params(**param)
    xgbc_pl.fit(concate_data[:-(validation_size+test_size), 2:167], concate_data[:-(validation_size+test_size), -1].astype(int))
    print("pseudo labeling train set")
    pred_labels = xgbc_pl.predict(concate_data[:-(validation_size+test_size), 2:167]) #train
    get_score(pred_labels, concate_data[:-(validation_size+test_size), -2:])
    """
    print("pseudo labeling validation set")
    pred_labels = xgbc_pl.predict(concate_data[-(validation_size+test_size):-test_size, 2:167]) #train
    get_score(pred_labels, concate_data[-(validation_size+test_size):-test_size, -2:])
    """
    print("pseudo labeling test set")
    pred_labels = xgbc_pl.predict(concate_data[-test_size:, 2:167]) #test
    get_score(pred_labels, concate_data[-test_size:, -2:])

    """#write label
    df_f = pd.read_csv(args.data_dir + "elliptic_txs_features.csv", header=None)
    feature = df_f.to_numpy()
    print(feature.shape)
    df_l = pd.read_csv(args.data_dir + "elliptic_txs_classes.csv", header=None)
    label = df_l.to_numpy()
    label = label[1:]
    count_1, count_2, count_uk = 0, 0, 0
    for i in range(0, feature.shape[0]):
      if int(feature[i, 0]) != int(label[i, 0]):
        print("not match txid!")
        print(f"{i},{int(feature[i, 0])},{int(label[i, 0])}")
        break
      if label[i, 1].strip() == '1':
        count_1 += 1
      elif label[i, 1].strip() == '2':
        count_2 += 1
      else:
        count_uk += 1
        pred_labels = xgbc.predict(feature[i, 2:167].reshape(-1, 165))
        df_l.at[i, 1] = pred_labels[0]
      if i % 10000 == 0:
        print(f"{i}/{feature.shape[0]}")
  
    df_l.to_csv(args.data_dir + "pseudo_label_classes.csv", index=False)
    print(count_1, count_2, count_uk)
    """


