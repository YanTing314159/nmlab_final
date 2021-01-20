#============================import================================
import os
import sys
import csv
import pandas as pd
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#============================loss and model========================
def get_loss(y , label , weight = [1,1,1], use_GPU=False):
  if use_GPU:
    weight = torch.Tensor(weight).to('cuda')
  else:
    weight = torch.Tensor(weight)
  criterion = nn.CrossEntropyLoss(weight)
  loss = criterion(y, label)
  return loss
def save_model(dir, model, opt, iter):
  iter=str(iter)
  if not os.path.exists(dir):
    os.mkdir(dir)
  if iter.isnumeric():
    iter=int(iter)
    if iter < 0:
      torch.save(model.state_dict(), f'{dir}/model.ckpt')
      torch.save(opt.state_dict(), f'{dir}/opt.opt')
    else:
      torch.save(model.state_dict(), f'{dir}/model.ckpt-{iter}')
      torch.save(opt.state_dict(), f'{dir}/opt.opt-{iter}')
  else:
    torch.save(model.state_dict(), f'{dir}/model.ckpt-{iter}')
    torch.save(opt.state_dict(), f'{dir}/opt.opt-{iter}')
def load_model(dir, model, opt, iter):
  iter=str(iter)
  if iter.isnumeric():
    iter=int(iter)
    if iter < 0:
      model.load_state_dict(torch.load(f'{dir}/model.ckpt'))
      opt.load_state_dict(torch.load(f'{dir}/opt.opt'))
    else:
      model.load_state_dict(torch.load(f'{dir}/model.ckpt-{iter}'))
      opt.load_state_dict(torch.load(f'{dir}/opt.opt-{iter}'))
  else:
    model.load_state_dict(torch.load(f'{dir}/model.ckpt-{iter}'))
    opt.load_state_dict(torch.load(f'{dir}/opt.opt-{iter}'))
  return model, opt
#============================dataset and loader====================
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
#============================models================================
class MLP_0(nn.Module):
     def __init__(self):
         self.name='MPL_0'
         super( MLP_0 ,self).__init__()
         self.layers=[]
         self. linear1 =torch.nn.Linear( 165 , 3 )
         self.layers.append(self. linear1 )
         self. act1 =torch.nn.Sigmoid()
         self.layers.append(self. act1 )
     def forward(self,x):
         o=x
         for i in self.layers:
             o=i(o)
         return o
class MLP_1(nn.Module):
     def __init__(self):
         super( MLP_1 ,self).__init__()
         self.name='MPL_1'
         self.layers=[]
         self. linear1 =torch.nn.Linear( 165 , 50 )
         self.layers.append(self. linear1 )
         self. linear2 =torch.nn.Linear( 50 , 3 )
         self.layers.append(self. linear2 )
         self. act1 =torch.nn.Sigmoid()
         self.layers.append(self. act1 )
     def forward(self,x):
         o=x
         for i in self.layers:
             o=i(o)
         return o
class MLP_2(nn.Module):
     def __init__(self):
         super( MLP_2 ,self).__init__()
         self.name='MPL_2'
         self.layers=[]
         self. linear1 =torch.nn.Linear( 165 , 80 )
         self.layers.append(self. linear1 )
         self. linear2 =torch.nn.Linear( 80 , 20 )
         self.layers.append(self. linear2 )
         self. act1 =torch.nn.ReLU()
         self.layers.append(self. act1 )
         self. linear3 =torch.nn.Linear( 20 , 3 )
         self.layers.append(self. linear3 )
         self. act2 =torch.nn.Sigmoid()
         self.layers.append(self. act2 )
     def forward(self,x):
         o=x
         for i in self.layers:
             o=i(o)
         return o
class MLP_3(nn.Module):
     def __init__(self):
         super( MLP_3 ,self).__init__()
         self.name='MPL_3'
         self.layers=[]
         self. linear1 =torch.nn.Linear( 165 , 80 )
         self.layers.append(self. linear1 )
         self. act1 =torch.nn.ReLU()
         self.layers.append(self. act1 )
         self. linear2 =torch.nn.Linear( 80 , 40 )
         self.layers.append(self. linear2 )
         self. act2 =torch.nn.ReLU()
         self.layers.append(self. act2 )
         self. linear3 =torch.nn.Linear( 40 , 20 )
         self.layers.append(self. linear3 )
         self. act3 =torch.nn.ReLU()
         self.layers.append(self. act3 )
         self. linear4 =torch.nn.Linear( 20 , 3 )
         self.layers.append(self. linear4 )
         self. act4 =torch.nn.Sigmoid()
         self.layers.append(self. act4 )
     def forward(self,x):
         o=x
         for i in self.layers:
             o=i(o)
         return o
class MLP_4(nn.Module):
     def __init__(self):
         super( MLP_4 ,self).__init__()
         self.name='MPL_4'
         self.layers=[]
         self. linear1 =torch.nn.Linear( 165 , 80 )
         self.layers.append(self. linear1 )
         self. linear2 =torch.nn.Linear( 80 , 40 )
         self.layers.append(self. linear2 )
         self. act1 =torch.nn.ReLU()
         self.layers.append(self. act1 )
         self. linear3 =torch.nn.Linear( 40 , 20 )
         self.layers.append(self. linear3 )
         self. act2 =torch.nn.ReLU()
         self.layers.append(self. act2 )
         self. linear4 =torch.nn.Linear( 20 , 3 )
         self.layers.append(self. linear4 )
         self. act3 =torch.nn.Sigmoid()
         self.layers.append(self. act3 )
     def forward(self,x):
         o=x
         for i in self.layers:
             o=i(o)
         return o
class MLP_5(nn.Module):
     def __init__(self):
         super( MLP_5 ,self).__init__()
         self.name='MPL_5'
         self.layers=[]
         self. linear1 =torch.nn.Linear( 165 , 40 )
         self.layers.append(self. linear1 )
         self. linear2 =torch.nn.Linear( 40 , 30 )
         self.layers.append(self. linear2 )
         self. act1 =torch.nn.ReLU()
         self.layers.append(self. act1 )
         self. linear3 =torch.nn.Linear( 30 , 25 )
         self.layers.append(self. linear3 )
         self. linear4 =torch.nn.Linear( 25 , 20 )
         self.layers.append(self. linear4 )
         self. act2 =torch.nn.ReLU()
         self.layers.append(self. act2 )
         self. linear5 =torch.nn.Linear( 20 , 15 )
         self.layers.append(self. linear5 )
         self. linear6 =torch.nn.Linear( 15 , 10 )
         self.layers.append(self. linear6 )
         self. act3 =torch.nn.ReLU()
         self.layers.append(self. act3 )
         self. linear7 =torch.nn.Linear( 10 , 5 )
         self.layers.append(self. linear7 )
         self. linear8 =torch.nn.Linear( 5 , 3 )
         self.layers.append(self. linear8 )
         self. act4 =torch.nn.Sigmoid()
         self.layers.append(self. act4 )
     def forward(self,x):
         o=x
         for i in self.layers:
             o=i(o)
         return o
#============================read data=============================
def read_inputs(data_folder,feature_name,class_name):
    df1 = pd.read_csv(data_folder+feature_name, header=None)
    data = df1.to_numpy()
    df2 = pd.read_csv(data_folder+class_name)
    label = df2.to_numpy()
    idx=[]
    lst_label=[]
    lst_data=[]
    for i in range(len(label)):    
      if data[i][0]!=label[i][0]:
        print('ERROR',data[i][0],label[i][0])
      if label[i][1]=='unknown':
        idx.append(i)
      else:
        lst_label.append(label[i])
        lst_data.append(data[i])
    print('before : ',label.shape)
    data=np.array(lst_data)
    label=np.array(lst_label)
    print('after : ',label.shape)
    norm_data = (data - data.min(0)) / data.ptp(0)
    return norm_data,label
def seperating_dataset(total_data,total_label,validation_rate,batch_num=4):
    size=int(total_data.shape[0]*validation_rate)
    train_dataset=myDataset(total_data[:-size],total_label[:-size])
    train_loader=DataLoader(train_dataset,batch_size=batch_num,shuffle=True,num_workers=2,drop_last=True)
    train_iterator=infinite_iter(train_loader)
    validation_dataset=myDataset(total_data[-size:],total_label[-size:])
    validation_loader=DataLoader(validation_dataset,batch_size=batch_num,shuffle=True,num_workers=2,drop_last=True)
    validation_iterator=infinite_iter(validation_loader)
    return (train_dataset,train_loader,train_iterator),(validation_dataset,validation_loader,validation_iterator)
def get_weight(train_label):
    return [0,8,1]
    label=[0,0,0]
    total=0
    for i in train_label:
      total=total+1
      if i[1] == 'unknown':
        label[0]=label[0]+1
      elif i[1] == '1':
        label[1]=label[2]+1
      elif i[1] == '2':
        label[2]=label[2]+1
    label[0]=1
    for i in range(1,len(label)):
      label[i]=total/label[i]
    return label
def training_process(model_path,train,validation,model,opt,iteration,suffix,use_GPU=False,log=False):
    max_f1=0
    train_dataset=train[0]
    train_loader=train[1]
    train_iterator=train[2]
    validation_dataset=validation[0]
    validation_loader=validation[1]
    validation_iterator=validation[2]
    weight=get_weight(train[0].label)
    model_dir=model_path+'model_'+str(suffix)
    log_dir=model_path+'model_'+str(suffix)
    if use_GPU:
        model = model.to('cuda')
    model.train()
    print('start training...')
    s_t=time.time()
    for iteration in range(0, total_iteration):
        x, y = next(train_iterator)
        if use_GPU:
            x = x.to('cuda')
            y = y.to('cuda')
        y_pred = model(x)
        loss = get_loss(y_pred, y.squeeze(),weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ((iteration+1) % 100 == 0):
          model.eval()
          #validation sample size
          val_sample = 100
          TP = 0
          TN = 0
          FP = 0
          FN = 0
          for _ in range(0, val_sample):
            x, y = next(validation_iterator)
            if use_GPU:
              x = x.to('cuda')
              y = y.to('cuda')
            pred = model(x)
            if _ == 0:
              val_loss = get_loss(pred, y.squeeze(), weight)
            else:
              _loss = get_loss(pred, y.squeeze(), weight)
              val_loss += _loss
            pred = pred.detach().cpu().numpy()
            pred_label = np.argmax(pred[:, 1:], axis=1)+1
            for _i in range(0, pred_label.shape[0]):
              if pred_label[_i] == 1 and y[_i].item() == 1:
                TP += 1
              elif pred_label[_i] == 2 and y[_i].item() == 1:
                FP += 1
              elif pred_label[_i] == 1 and y[_i].item() == 2:
                FN += 1
              elif pred_label[_i] == 2 and y[_i].item() == 2:
                TN += 1
              else:
                print("unexcept!")
                print(y[_i].item())
                break
          val_loss /= val_sample
          model.train()
          et = time.time() - s_t
          et = str(datetime.timedelta(seconds=et))[:-7]
          #print training process
          Recall = TP/(TP+FN+1e-6)
          Precision = TP/(TP+FP+1e-6)
          F1score = 2 * Precision * Recall / (Precision + Recall+1e-6)
          if F1score>max_f1:
            print('saving by f1 : ',F1score)
            save_model(model_dir, model, opt, '_max')
            max_f1=F1score
          print("[{}]iteration {} train loss: {}, val loss: {}, f1:{}".format(et, iteration+1, loss.item(), val_loss.item(), F1score))
          if log:
            logger.add_scalar('train loss', loss, iteration+1)
            logger.add_scalar('val loss', val_loss, iteration+1)
            logger.add_scalar('F1 score', F1score, iteration+1)
        if ((iteration+1) % 100 == 0):
            save_model(model_dir, model, opt, -1)
        if ((iteration+1) % 1000 == 0):
            save_model(model_dir, model, opt, iteration+1)
    return
def predict_by_Ensemble(models,features):
    nums=0
    x=features.T[2:]
    x=x.astype('float32')
    x=torch.from_numpy(x.T)
    for i in models:
        i.eval()
        i.to('cpu')
        if nums==0:
            ans=i(x)
        else:
            ans=ans+i(x)
        nums=nums+1
    ans/=nums
    return ans
def predict_by_Ensemble_vote(models,features):
    nums=len(models)
    x=features.T[2:]
    x=x.astype('float32')
    x=torch.from_numpy(x.T)
    num=0
    ans=np.zeros((features.shape[0],3))
    for i in models:
        i.eval()
        i.to('cpu')
        temp=i(x).detach().numpy()
        for j in temp:
          k=j.argmax()
          j[0]=0
          j[1]=0
          j[2]=0
          j[k]=1
        ans=ans+temp
    return ans
def opt_generator(model):
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    return opt
def get_f1_score(predicted,validation_data):
  ans=[]
  wrong=0
  total=0
  valid=[0,0,0]
  predicted_label=[0,0,0]
  tp=0
  fp=0
  fn=0
  for i in predicted:
    ans.append(int(i.argmax()))
    predicted_label[int(i.argmax())]=predicted_label[int(i.argmax())]+1
  for i in range(validation_data[0].label.shape[0]):
    total=total+1
    if validation_data[0].label[i][1]=='unknown':
      valid[0]=valid[0]+1
    elif validation_data[0].label[i][1]=='1':
      valid[1]=valid[1]+1
    elif validation_data[0].label[i][1]=='2':
      valid[2]=valid[2]+1
    if ans[i]==0 and validation_data[0].label[i][1]=='1':
      wrong=wrong+1
    elif ans[i]==0 and validation_data[0].label[i][1]=='2':
      wrong=wrong+1
    elif ans[i]==1 and validation_data[0].label[i][1]=='unknown':
      wrong=wrong+1
    elif ans[i]==1 and validation_data[0].label[i][1]=='1':
      tp=tp+1
    elif ans[i]==1 and validation_data[0].label[i][1]=='2':
      wrong=wrong+1
      fn=fn+1
    elif ans[i]==2 and validation_data[0].label[i][1]=='unknown':
      wrong=wrong+1
    elif ans[i]==2 and validation_data[0].label[i][1]=='1':
      wrong=wrong+1
      fp=fp+1
  precision=tp/(tp+fp+1e-6)
  recall=tp/(tp+fn+1e-6)
  f1=2*precision*recall/(precision+recall+1e-6)
  return f1
#============================data name=============================
class_name="elliptic_txs_classes.csv"
edgelist_name="elliptic_txs_edgelist.csv"
feature_name="elliptic_txs_features.csv"
#============================main==================================
if __name__=='__main__':
    if len(sys.argv)!=4:
        print('usage : python MLP.py <dataset path> <model path> <mode (\'-training\' , \'-testing\')>')
        exit(1)
    if sys.argv[3]=='-training':
        is_training=True
    elif sys.argv[3]=='-testing':
        is_training=False
    else:
        print('usage : python MLP.py <> <> <>')
        exit(1)
    data_folder=sys.argv[1]
    model_path=sys.argv[2]
    if not os.path.exists(model_path):
        os.mkdir(dir)
    total_iteration,validation_rate,batch_num=50000,0.3,8
    models=[MLP_0(),MLP_1(),MLP_2(),MLP_3(),MLP_4(),MLP_5()]
    opts=[]
    suffix=[]
    for i in range(len(models)):
      suffix.append(str(i))
      opts.append(opt_generator(models[i]))
    norm_data,label=read_inputs(data_folder,feature_name,class_name)
    train_data,validation_data=seperating_dataset(norm_data,label,validation_rate,batch_num)
    if is_training:
      for i in range(len(suffix)):
        training_process(model_path,train_data,validation_data,models[i],opts[i],total_iteration,suffix[i])
    else:
      for i in suffix:
        i=int(i)
        model,opt=load_model(model_path+'model_'+suffix[i],models[i],opts[i],'_max')

    for model in models:
      temp=[model]
      predicted_valid=predict_by_Ensemble(temp,validation_data[0].data)
      f1_valid=get_f1_score(predicted_valid,validation_data)
      predicted_train=predict_by_Ensemble(temp,train_data[0].data)
      f1_train=get_f1_score(predicted_train,train_data)
      print(model.name,' train ',f1_train, ', test ',f1_valid)