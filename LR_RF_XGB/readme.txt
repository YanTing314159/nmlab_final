Logistic Regression：
  requirement：
    python 3.7
    pytorch 1.7
  usage：
    bash LR.sh
    or
    python ./src/logistic_regression.py -flag test -output_model_dir <model dir> -load_model_dir ./model_LR -data_dir <data dir>
    when flag is test, this will load model from ./model_LR and output test result.
    when flag is train, this will train a new model and save it in <model dir>

Random Forest：
  requirement：
    python 3.7
    scikit-learn 0.24.0
  usage：
    bash RF.sh
    or
    python ./src/random_forest.py -data_dir <data dir>
    this will construct a random forest and output the testing f1 score

XGBoost：
  requirement：
    python 3.7
    scikit-learn 0.24.0
  usage：
    bash RF.sh
    or
    python ./src/random_forest.py -data_dir <data dir>
    this will construct a XGBoost model and output the testing f1 score

