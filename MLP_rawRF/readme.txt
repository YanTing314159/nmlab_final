MLP.py:
  python MLP.py <dataset path(elliptic_bitcoin_dataset)> <model path(where to save and load the model)> <mode('-training','-testing')>
  This would train or load model from model path and test on data in dataset path.
  dataset path should contain : elliptic_txs_classes.csv elliptic_txs_edgelist.csv elliptic_txs_features.csv Result.csv

reading.py:
  python reading.py <data path> <mode((-statistics/-load))>
  This would calculate the data and raw data in the data path and show the ave and std of the features and raw datas, and plot the data in data path/feature_pics/ and data path/raw_pics/.
  data path should contain : elliptic_bitcoin_dataset/* raw_npy/* statistic_data/*

training_by_raw.py:
  python train_by_raw.py <data path> <training method(-xgboost or -randomforest)>
  This would train by raw data by xgboost or randomforest.
  data path should contain elliptic_bitcoin_dataset/* , raw_npy/* , txs/* , txs is the folder that contains the raw transactions from ta's tar file by tar xvf txs.tar

In short the file structure should be:
NMLab-Final/
  code/
    MLP.py
    reading.py
    train_by_raw.py
  data/
    *** elliptic_bitcoin_dataset/ (This file is from https://drive.google.com/drive/folders/1hQQPQ81mZPkU4Fpw_j-5l2aRvTAk5HGq)
    raw_npy/
    statistic_data/
    *** txs/  (This file is too large,please use tar xvf txs.tar to get this and put all json files into txs/ here)
    *** : need to add
  models/
    model_0/
    model_1/
    model_2/
    model_3/
    model_4/
    model_5/
  readme.txt