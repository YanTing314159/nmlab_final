import numpy as np
import argparse
from data_utils import preprocess_dropuk
from utils import get_score
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-data_dir', default='./data/')

  args = parser.parse_args()
  #====load data
  clean_data, clean_label, uk_data = preprocess_dropuk(args.data_dir, norm=False, return_uk=True)
  concate_data = np.c_[clean_data.reshape(len(clean_data), -1), clean_label[:,1].reshape(len(clean_label), -1)].astype(int)
  #print(concate_data.shape) #(46564, 168)
  np.random.seed(0)

  validation_split_rate = 0.0
  validation_size = int(concate_data.shape[0]*validation_split_rate)
  test_split_rate = 0.3
  test_size = int(concate_data.shape[0]*test_split_rate)
  print("building random forest...")
  clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0, class_weight={1:1,2:1}, bootstrap=True, max_features=10)
  clf.fit(concate_data[:-(validation_size+test_size), 2:167], concate_data[:-(validation_size+test_size), -1].astype(int))
  print("train set")
  pred_labels = clf.predict(concate_data[:-(validation_size+test_size), 2:167]) #train
  get_score(pred_labels, concate_data[:-(validation_size+test_size), -2:])
  #print("validation set")
  #pred_labels = clf.predict(concate_data[-(validation_size+test_size):-test_size, 2:167]) #validation
  #get_score(pred_labels, concate_data[-(validation_size+test_size):-test_size, -2:])
  print("test set")
  pred_labels = clf.predict(concate_data[-test_size:, 2:167]) #test
  get_score(pred_labels, concate_data[-test_size:, -2:])

  importances = clf.feature_importances_
  indices = np.argsort(importances)[::-1]
  print("==================")
  print(importances)
  print("==================")
  print(indices)
  #pusedo labeling
  #"""
  pred_labels = clf.predict(uk_data[:, 2:167])
  #print(pred_labels.shape) #(157205,)
  #print(uk_data.shape) #(157205, 167)
  concate_uk_data = np.c_[uk_data.reshape(len(uk_data), -1), pred_labels.reshape(len(pred_labels), -1)].astype(int)
  #print(concate_uk_data.shape) #(157205, 168)
  concate_data = np.concatenate((concate_uk_data, concate_data), axis=0)
  #print(concate_data.shape)#(157205, 168) + (46564, 168) = (203769, 168)
  #pseudo labeling
  print("building random forest with pseudo labeling...")
  clf_pl = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0, class_weight={1:1,2:1}, bootstrap=True, max_features=50)
  clf_pl.fit(concate_data[:-(validation_size+test_size), 2:167], concate_data[:-(validation_size+test_size), -1].astype(int))
  print("pseudo labeling train set")
  pred_labels = clf_pl.predict(concate_data[:-(validation_size+test_size), 2:167]) #train
  get_score(pred_labels, concate_data[:-(validation_size+test_size), -2:])
  #print("pseudo labeling validation set")
  #pred_labels = clf_pl.predict(concate_data[-(validation_size+test_size):-test_size, 2:167]) #test
  #get_score(pred_labels, concate_data[-(validation_size+test_size):-test_size, -2:])
  print("pseudo labeling test set")
  pred_labels = clf_pl.predict(concate_data[-test_size:, 2:167]) #test
  get_score(pred_labels, concate_data[-test_size:, -2:])
  #"""




