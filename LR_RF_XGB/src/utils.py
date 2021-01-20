import os
import numpy as np
import torch
import torch.nn as nn

#loss function
def weighted_CEloss(y, label, weight):
  criterion = nn.CrossEntropyLoss(weight)
  loss = criterion(y, label)
  return loss

#model save, load
def save_model(dir, model, opt, iter):
  if not os.path.exists(dir):
    os.makedirs(dir)
  if str(iter).isnumeric():
    iter = int(iter)
    if iter <= 0:
      torch.save(model.state_dict(), f'{dir}/model.ckpt')
      torch.save(opt.state_dict(), f'{dir}/opt.opt')
    else:
      torch.save(model.state_dict(), f'{dir}/model.ckpt-{iter}')
      torch.save(opt.state_dict(), f'{dir}/opt.opt-{iter}')
  else:
    torch.save(model.state_dict(), f'{dir}/model.ckpt-{iter}')
    torch.save(opt.state_dict(), f'{dir}/opt.opt-{iter}')

def load_model(dir, model, opt, iter):
  if str(iter).isnumeric():
    iter = int(iter)
    if iter <= 0:
      model.load_state_dict(torch.load(f'{dir}/model.ckpt'))
      opt.load_state_dict(torch.load(f'{dir}/opt.opt'))
    else:
      model.load_state_dict(torch.load(f'{dir}/model.ckpt-{iter}'))
      opt.load_state_dict(torch.load(f'{dir}/opt.opt-{iter}'))
  else:
    torch.save(model.state_dict(), f'{dir}/model.ckpt-{iter}')
    torch.save(opt.state_dict(), f'{dir}/opt.opt-{iter}')
  return model, opt

#NN model evaluater
def NN_evaluater(model, loss_weight, data_iterator, sample_num, use_gpu): #for single model
  TP = 0
  TN = 0
  FP = 0
  FN = 0
  for _ in range(0, sample_num):
    x, y = next(data_iterator)
    if use_gpu:
      x = x.to('cuda')
      y = y.to('cuda')
    pred = model(x)
    if _ == 0:
      val_loss = weighted_CEloss(pred, y.squeeze(), loss_weight)
    else:
      _loss = weighted_CEloss(pred, y.squeeze(), loss_weight)
      val_loss += _loss
    pred = pred.detach().numpy()
    pred_label = np.argmax(pred[:, 1:], axis=1)+1
    for _i in range(0, pred_label.shape[0]):
      if pred_label[_i] == 1 and y[_i].item() == 1:
        TP += 1
      elif pred_label[_i] == 1 and y[_i].item() == 2:
        FP += 1
      elif pred_label[_i] == 2 and y[_i].item() == 1:
        FN += 1
      elif pred_label[_i] == 2 and y[_i].item() == 2:
        TN += 1
      else:
        print("unexcept!")
        print(y[_i].item())
        break
  val_loss /= sample_num
  #print training process
  Recall = TP/(TP+FN+1e-6)
  Precision = TP/(TP+FP+1e-6)
  F1score = 2 * Precision * Recall / (Precision + Recall+1e-6)
  """
  print("TP:", TP, "FN:", FN)
  print("FP:", FP, "TN:", TN)
  print("Recall:", Recall)
  print("Precision:", Precision)
  print("F1score:", F1score)
  """
  return val_loss, Recall, Precision, F1score

def NN_ensemble_evaluater(model, dataset, use_gpu, mode, result, end): #for ensemble
  for i in range(0, dataset.__len__()):
    x, y = dataset.__getitem__(i)
    if use_gpu:
      x = x.to('cuda')
      y = y.to('cuda')
    pred = model(x)
    pred = pred.detach().numpy()
    pred_label = np.argmax(pred[1:])
    if mode == 'ave':
      result[i] += pred[1:]
    if mode == 'vote':
      result[i, pred_label] += 1
    
  if end:
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    result = np.argmax(result[:], axis=1)
    result = result + np.ones(result.shape[0])
    for i in range(0, dataset.__len__()):
      x, y = dataset.__getitem__(i)
      if result[i] == 1 and y.item() == 1:
        TP += 1
      elif result[i] == 1 and y.item() == 2:
        FP += 1
      elif result[i] == 2 and y.item() == 1:
        FN += 1
      elif result[i] == 2 and y.item() == 2:
        TN += 1
      else:
        print("unexcept!")
        print(y.item())
        break
    Recall = TP/(TP+FN+1e-6)
    Precision = TP/(TP+FP+1e-6)
    F1score = 2 * Precision * Recall / (Precision + Recall+1e-6)
    """
    print("TP:", TP, "FN:", FN)
    print("FP:", FP, "TN:", TN)
    print("Recall:", Recall)
    print("Precision:", Precision)
    print("F1score:", F1score)
    """
    return Recall, Precision, F1score
  else:
    return result

def get_score(pred_labels, clean_label): #for random forest
  TP = 0
  TN = 0
  FP = 0
  FN = 0
  pred_count = {}
  label_count = {}
  for i in range(0, len(pred_labels)):
    pred_label, y = int(pred_labels[i]), int(clean_label[i, 1])
    if pred_label not in pred_count:
      pred_count[pred_label] = 0
    else:
      pred_count[pred_label] += 1
    if y not in label_count:
      label_count[y] = 0
    else:
      label_count[y] += 1
    if pred_label == 1 and y == 1:
      TP += 1
    elif pred_label == 2 and y == 1:
      FP += 1
    elif pred_label == 1 and y == 2:
      FN += 1
    elif pred_label == 2 and y == 2:
      TN += 1
    else:
      print("unexcept!")
  print("pred_count:", pred_count)
  print("label_count:", label_count)
  print("TP:", TP, "FN:", FN)
  print("FP:", FP, "TN:", TN)

  Recall = TP/(TP+FN+1e-8)
  Precision = TP/(TP+FP+1e-8)
  F1score = 2 * Precision * Recall / (Precision + Recall+1e-8)
  print("Recall:", Recall)
  print("Precision:", Precision)
  print("F1score:", F1score)

def VAE_evaluater(model, criterion, loss_threshold, lamda_kl, data_iterator, sample_num, use_gpu):
  pred_labels = []
  clean_label = []
  for i in range(0, sample_num):
    x, y = next(data_iterator)
    if use_gpu:
      x = x.to('cuda')
      y = y.to('cuda')
    x_hat, kld = model(x)
    loss_rec = criterion(x_hat, x)
    if kld is not None:
      elbo = loss_rec + lamda_kl * kld
      loss = elbo
    else:
      print("train kld is none! :", iteration+1)
    if y.detach().cpu().numpy()[0] == 1:
      print("1 loss:", loss.item(), "rec:", loss_rec.item(), "kld:", kld.item())
    elif y.detach().cpu().numpy()[0] == 2:
      print("2 loss:", loss.item(), "rec:", loss_rec.item(), "kld:", kld.item())
    else:
      print("???????")
    if use_gpu:
      clean_label.append([0, y.detach().cpu().numpy()[0]])
    else:
      clean_label.append([0, y.detach().numpy()[0]])
    if loss.item() > loss_threshold:
      pred_labels.append(2)
    else:
      pred_labels.append(1)
  clean_label = np.array(clean_label)
  get_score(pred_labels, clean_label)

def try_(r):
  r += np.ones(2)

