import time
import datetime
import torch
import torch.nn as nn
import numpy as np
from data_utils import myDataset, get_data_iterater, preprocess_dropuk
from data_utils import get_data_iterater
from utils import save_model, load_model, weighted_CEloss, NN_evaluater
from model_LR import LogisticRegression
from tensorboardX import SummaryWriter
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-flag', default='train')
  parser.add_argument('-output_model_dir', default='./model_LR')
  parser.add_argument('-load_model_dir', default='./model_LR')
  parser.add_argument('-data_dir', default='./data/')
    
  args = parser.parse_args()
  #====load data
  norm_data, clean_label = preprocess_dropuk(args.data_dir)
  #====dataset split rate
  validation_split_rate = 0.2
  validation_size = int(norm_data.shape[0]*validation_split_rate)
  test_split_rate = 0.2
  test_size = int(norm_data.shape[0]*test_split_rate)
  #====train
  if args.flag == 'train':
    batch_size = 64
    shuffle = True
    total_iteration = 10000
    use_gpu = False
    log = True #require tensorboardX
    log_dir = './log_LR'

    train_dataset = myDataset(norm_data[:-(validation_size+test_size)],
            clean_label[:-(validation_size+test_size)])
    train_iterator = get_data_iterater(train_dataset, batch_size, shuffle)
    validation_dataset = myDataset(norm_data[-(validation_size+test_size):-test_size],
            clean_label[-(validation_size+test_size):-test_size])
    validation_iterator = get_data_iterater(validation_dataset, batch_size, shuffle)

    model = LogisticRegression()
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=0)
    if use_gpu:
      model = model.to('cuda')
    if log:
      logger = SummaryWriter(log_dir)
    max_validation_f1 = 0
    
    model.train()
    print("start training")
    s_t = time.time()
    for iteration in range(0, total_iteration):
      x, y = next(train_iterator)
      if use_gpu:
        x = x.to('cuda')
        y = y.to('cuda')
      y_pred = model(x)
      weight = torch.Tensor([0, 20, 1]) #weight for each class (unknown, illict, licit)
      loss = weighted_CEloss(y_pred, y.squeeze(), weight)
      opt.zero_grad()
      loss.backward()
      opt.step()
      if ((iteration+1) % 100 == 0):
        model.eval()
        #NN_evaluater(model, loss_weight, data_iterater, sample_num, use_gpu)
        sample_num = 100
        weight = torch.Tensor([0, 8, 1])
        val_loss, Recall, Precision, val_F1score = NN_evaluater(model, weight, validation_iterator, sample_num, use_gpu)
        if val_F1score > max_validation_f1:
          save_model(args.output_model_dir, model, opt, 'max')
          max_validation_f1 = val_F1score
        _, _, _, train_F1score = NN_evaluater(model, weight, train_iterator, sample_num, use_gpu)
        model.train()
        #print training process
        et = time.time() - s_t
        et = str(datetime.timedelta(seconds=et))[:-7]
        print("[{}]iteration {} train loss: {:.6f}, val loss: {:.6f}, train F1:{:.6f} val F1:{:.6f}"
           .format(et, iteration+1, loss.item(), val_loss.item(), train_F1score, val_F1score))
        if log:
          logger.add_scalar('train loss', loss, iteration+1)
          logger.add_scalar('val loss', val_loss, iteration+1)
          logger.add_scalar('train F1 score', train_F1score, iteration+1)
          logger.add_scalar('val F1 score', val_F1score, iteration+1)
      if ((iteration+1) % 2000 == 0):
        save_model(args.output_model_dir, model, opt, 0)
      if ((iteration+1) % 10000 == 0):
        save_model(args.output_model_dir, model, opt, iteration+1)
    print("finish")
    print("max validation f1: ", max_validation_f1)
  #====test
  if args.flag == 'test':
    batch_size = 128
    shuffle = False
    use_gpu = False

    train_dataset = myDataset(norm_data[:-(validation_size+test_size)],
            clean_label[:-(validation_size+test_size)])
    train_iterator = get_data_iterater(train_dataset, batch_size, shuffle, drop_last=False)
    validation_dataset = myDataset(norm_data[-(validation_size+test_size):-test_size],
            clean_label[-(validation_size+test_size):-test_size])
    validation_iterator = get_data_iterater(validation_dataset, batch_size, shuffle, drop_last=False)

    test_dataset = myDataset(norm_data[-test_size:], clean_label[-test_size:])
    test_iterator = get_data_iterater(test_dataset, batch_size, shuffle, drop_last=False)

    model = LogisticRegression()
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    model, opt = load_model(args.load_model_dir, model, opt, 10000)
    model.eval()

    sample_num = int(train_dataset.__len__()/batch_size)+1
    weight = torch.Tensor([0, 8, 1])
    _, Recall, Precision, F1score = NN_evaluater(model, weight, train_iterator, sample_num, use_gpu)
    print("train")
    print("Recall:", Recall)
    print("Precision:", Precision)
    print("F1score:", F1score)

    sample_num = int(validation_dataset.__len__()/batch_size)+1
    weight = torch.Tensor([0, 8, 1])
    _, Recall, Precision, F1score = NN_evaluater(model, weight, validation_iterator, sample_num, use_gpu)
    print("validation")
    print("Recall:", Recall)
    print("Precision:", Precision)
    print("F1score:", F1score)

    sample_num = int(test_dataset.__len__()/batch_size)+1
    weight = torch.Tensor([0, 8, 1])
    _, Recall, Precision, F1score = NN_evaluater(model, weight, test_iterator, sample_num, use_gpu)
    print("test")
    print("Recall:", Recall)
    print("Precision:", Precision)
    print("F1score:", F1score)

