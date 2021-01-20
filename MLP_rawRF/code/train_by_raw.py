import os
import sys
import time
import json
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from collections import OrderedDict
import xgboost as xgb

def constant_setting(folder_name):
    mode='load'
    folder_path=folder_name#'./NMLab-Final/data/'
    txs_path=folder_path+'txs/'
    id_2_txs_filename=folder_path+'elliptic_bitcoin_dataset/'+'Result.csv'
    class_file=folder_path+'elliptic_bitcoin_dataset/'+'elliptic_txs_classes.csv'
    feature_file=folder_path+'elliptic_bitcoin_dataset/'+'elliptic_txs_features.csv'
    construct_dict_name=txs_path+'0000e773d1bf4d4687029479fe1522a100f0bab19c1e1756939acef27a5c0a5c.json'
    npy_folder_name=folder_path+'statistic_data/'
    raw_pic_folder=folder_path+'raw_pics/'
    feature_pic_folder=folder_path+'feature_pics/'
    raw_npy_folder=folder_path+'raw_npy/'
    low_rate=2
    high_rate=98
    dummy=[]
    validation_test_rate=[0.2,0.2]
    for i in range(93,167):
        dummy.append(i)
    return mode,folder_path,txs_path,id_2_txs_filename,class_file,feature_file,dummy,validation_test_rate,npy_folder_name,raw_pic_folder,feature_pic_folder,low_rate,high_rate,raw_npy_folder
def parsing_not_labeled(feature,label,dummy):
    to_remove=[]
    new_label=[]
    label=np.delete(arr=label,obj=[0],axis=0)
    for i in range(len(label)):
        label[i][0]=int(label[i][0])
        if label[i][1]=='unknown':
            to_remove.append(i)
        else:
            label[i][1]=int(label[i][1])
    label=np.delete(arr=label,obj=to_remove,axis=0)
    label=label.astype(int)
    feature=np.delete(arr=feature,obj=dummy,axis=1)
    feature=np.delete(arr=feature,obj=to_remove,axis=0)
    return feature,label
def create_dicts(raw_id2txs,label):
    id2txs={}
    txs2id={}
    label_dict={}
    for i in label:
        label_dict[i[0]]=i[1]
    raw_id2txs=raw_id2txs[1:]
    for i in raw_id2txs:
        if label_dict.get(int(i[0]))!=None:
            id2txs[i[0]]=i[1]
            txs2id[i[1]]=i[0]
    return id2txs,txs2id
def seperating_illicit(training_features,training_label):
    illicit_set=[]
    licit_set=[]
    for i in range(training_label.shape[0]):
        if int(training_label[i][1])==1:
            illicit_set.append(i)
        else:
            licit_set.append(i)
    illicit_feature=training_features.copy()
    licit_feature=training_features.copy()
    illicit_feature=np.delete(arr=illicit_feature,obj=licit_set,axis=0)
    licit_feature=np.delete(arr=licit_feature,obj=illicit_set,axis=0)
    return (illicit_set,illicit_feature),(licit_set,licit_feature)
def seperating_validation(features,label,rates):
    total_num=label.shape[0]
    train_num=int(total_num*(1-rates[0]-rates[1]))
    valid_num=int(total_num*rates[0])
    test_num=total_num-train_num-valid_num
    return (features[0:train_num],label[0:train_num]),(features[train_num:train_num+valid_num],label[train_num:train_num+valid_num]),(features[train_num+valid_num:],label[train_num+valid_num:])
def training_random_forest(train_data):
    print('training by random forest ...')
    clf=RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0, class_weight={1:0.75,2:0.25}, bootstrap=True, max_features=8)
    feature=train_data[0]
    label=train_data[1]
    clf.fit(feature,label)
    return clf
def training_xgboost(train_data):
    print('training by xgboost ...')
    param = {}
    param['booster'] = 'gbtree'
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.3
    param['gamma'] = 0
    param['max_depth'] = 10
    param['min_child_weight']=1
    param['max_delta_step'] = 0
    param['subsample']= 1
    param['colsample_bytree']=1
    param['silent'] = 1
    param['seed'] = 0
    param['base_score'] = 0.5
    clf = xgb.XGBClassifier()
    clf.set_params(**param)
    clf.fit(train_data[0],train_data[1])
    return clf
def get_score(predict,ans):
    recall=0
    precision=0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(ans)):
        if predict[i]==1 and ans[i]==1:
            TP=TP+1
        elif predict[i]==2 and ans[i]==1:
            FP=FP+1
        elif predict[i]==1 and ans[i]==2:
            FN=FN+1
        elif predict[i]==2 and ans[i]==2:
            TN=TN+1
    Recall = TP/(TP+FN+1e-6)
    Precision = TP/(TP+FP+1e-6)
    F1score = 2 * Precision * Recall / (Precision + Recall+1e-6)
    return F1score
def get_a_Odict(folder_name,idtxs_table,hash_id):
    hash_id=str(int(hash_id))
    if idtxs_table.get(hash_id)==None:
        return None
    file_name=folder_name+str(idtxs_table[hash_id])+'.json'
    with open(file_name,'r') as f:
        raw_data=json.load(f, object_pairs_hook=OrderedDict)
    return raw_data
def normalize_training(training,folder='./'):
    temp=training.T
    first=True
    for i in temp:
        k=np.isnan(i)
        for j in range(k.shape[0]):
            if k[j] == True:
                i[j]=0
        feedback=i.sum()/i.shape[0]
        for j in range(k.shape[0]):
            if k[j] == True:
                i[j]=feedback
        base=i.max()-i.min()
        if base != 0:
            i=i/base
        if first:
            norm_data=np.array(i).reshape(-1,1)
            first=False
        else:
            i=np.array(i).reshape(-1,1)
            norm_data=np.concatenate((norm_data,i),axis=1)
    np.save(folder+'normed_raw_feature.npy',norm_data)
    return norm_data
def construct_raw_data_training_set(txs_path,id2txs,training_set,training_label,folder='./'):
    keys=[ 'weight', 'block_height',  'lock_time', 'size', 'block_index', 
          'tx_index', 'vin_sz', 'vout_sz']#'ver',, 'inputs', 'outs']
    inouts=['num_txin', 'num_txout', 'num_prev_out', 'value']
    first=True
    cant_find=0
    total_num=0
    labels=[]
    for idx in range(training_set.shape[0]):
        i=training_set[idx]
        now_label=training_label[idx]
        if(total_num%100)==0:
            print('processing ','can\'t find : ',cant_find,' , ',total_num,'/',len(training_set),' ...')
        total_num=total_num+1            
        hash_id=i[0]
        now_dict=get_a_Odict(txs_path,id2txs,hash_id)
        if now_dict==None:
            cant_find=cant_find+1
            continue
        else:
            row=[]
            labels.append(now_label[1])
            for item in keys:
                temp=now_dict.get(item)
                if temp==None:
                    print('data index not found')
                    temp=0
                row.append(temp)
            row=np.array(row).reshape(-1,1)
            if first:
                pool=row
                first=False
            else:
                pool=np.concatenate((pool,row),axis=1)
    pool=pool.T
    np.save(folder+'raw_data.npy',pool)
    np.save(folder+'raw_labels.npy',labels)
    return pool,labels
def load_npy_data(folder_name,low_rate=2,high_rate=98):
    part=['illicit','licit']
    keys=['height','lock_time','size','txind','ver','visize','vosize','weight']
    illicit_sets=[]
    licit_sets=[]
    for key in keys:
        ori_ill=np.load(folder_name+part[0]+'_'+key+'.npy')
        ori_lic=np.load(folder_name+part[1]+'_'+key+'.npy')
        ill=[]
        lic=[]
        ill_bound=[np.percentile(ori_ill,low_rate),np.percentile(ori_ill,high_rate)]
        lic_bound=[np.percentile(ori_lic,low_rate),np.percentile(ori_lic,high_rate)]
        for i in ori_ill:
            if i <= ill_bound[1] and i >= ill_bound[0]:
                ill.append(i)
        for i in ori_lic:
            if i <= lic_bound[1] and i >= lic_bound[0]:
                lic.append(i)
        illicit_sets.append(np.array(ill))
        licit_sets.append(np.array(lic))
    return illicit_sets,licit_sets
def draw_data(folder_name,illicit,licit):
    if os.path.isdir(folder_name)==False:
        os.makedirs(folder_name)
    keys=['height','lock_time','size','txind','ver','visize','vosize','weight']
    datas=[illicit,licit]
    y_cor=[100,200]
    for i in range(len(keys)):
        plt.figure()
        for j in [0,1]:
            data=datas[j][i]
            y=[]
            for k in range(len(data)):
                y.append(y_cor[j])
            plt.scatter(data,y,s=2)
        plt.savefig(folder_name+keys[i]+'.png')
        plt.close()
    return
def draw_feature(folder_name,illicit,licit):
    if os.path.isdir(folder_name)==False:
        os.makedirs(folder_name)
    datas=[illicit,licit]
    y_cor=[100,200]
    for i in range(len(illicit)):
        plt.figure()
        for j in [0,1]:
            data=datas[j][i]
            y=[]
            for k in range(len(data)):
                y.append(y_cor[j])
            plt.scatter(data,y,s=2)
        plt.savefig(folder_name+'feature_'+'0'*(3-len(str(i)))+str(i)+'.png')
        plt.close()
    return
def print_data_info(illicit,licit):
    keys=['height','lock_time','size','txind','ver','visize','vosize','weight']
    dict_name=['illicit','licit']
    datas=[illicit,licit]
    for j in [0,1]:
        data=datas[j]
        print(dict_name[j],' :')
        for i in range(len(keys)):
            string=keys[i]
            string=string+' '*(12-len(string))
            print(string,' ave : ',data[i].sum()/data[i].shape[0],' , std : ',data[i].std())
    return
def print_feature_info(illicit,licit):
    dict_name=['illicit','licit']
    datas=[illicit,licit]
    for i in [0,1]:
        data=datas[i]
        print(dict_name[i],' :')
        for j in range(len(data)):
            temp=np.array(data[j])
            if temp.shape[0]==0:
                ave=0
            else:
                ave=temp.sum()/temp.shape[0]
            print('feature'+'0'*(3-len(str(j)))+str(j),' , ave : ',ave,' std : ',temp.std())
    return
def filt_feature(illicit,licit,low_rate=2,high_rate=98):
    ill=illicit[1].copy()
    lic=licit[1].copy()
    ill_feature=[]
    lic_feature=[]
    ill_bound=[]
    lic_bound=[]
    ill=ill.T[1:]
    lic=lic.T[1:]
    for i in ill:
        ill_bound.append((np.percentile(i,low_rate),np.percentile(i,high_rate)))
    for i in lic:
        lic_bound.append((np.percentile(i,low_rate),np.percentile(i,high_rate)))
    ill=ill.tolist()
    lic=lic.tolist()
    for i in range(len(ill)):
        to_insert=[]
        for j in range(len(ill[i])):
            if ill[i][j]<=ill_bound[i][1] and ill[i][j]>=ill_bound[i][0]:
                to_insert.append(ill[i][j])
        ill[i]=to_insert.copy()
    for i in range(len(lic)):
        to_insert=[]
        for j in range(len(lic[i])):
            if lic[i][j]<=lic_bound[i][1] and lic[i][j]>=lic_bound[i][0]:
                to_insert.append(lic[i][j])
        lic[i]=to_insert.copy()
    return ill,lic

if __name__=='__main__':
    ConsTruct_raw_Data=False
    if len(sys.argv)!=3:
        print('usage: python train_by_raw.py [data path] [training method(-xgboost or -randomforest)]')
        exit(-1)
    if sys.argv[2]!='-xgboost' and sys.argv[2]!='-randomforest':
        print('usage: python train_by_raw.py [data path] [training method(-xgboost or -randomforest)]')
        exit(1)
    train_method=sys.argv[2]
    mode,folder_path,txs_path,id_2_txs_filename,class_file,feature_file,dummy,validation_test_rate,npy_folder,raw_pic_folder,feature_pic_folder,low_rate,high_rate,raw_npy_folder=constant_setting(sys.argv[1])
    raw_id2txs=pd.read_csv(id_2_txs_filename,header=None).to_numpy()
    training_features=pd.read_csv(feature_file,header=None).to_numpy()
    training_label=pd.read_csv(class_file,header=None).to_numpy()
    training_features,training_label=parsing_not_labeled(training_features,training_label,dummy)
    id2txs,txs2id=create_dicts(raw_id2txs,training_label)
    #=================================================================================================
    #illicit_data,licit_data=seperating_illicit(training_features,training_label)
    #illicit_feature,licit_feature=filt_feature(illicit_data,licit_data)
    if ConsTruct_raw_Data:
        raw_training,raw_label=construct_raw_data_training_set(txs_path,id2txs,training_features,training_label,raw_npy_folder)
        normed=normalize_training(raw_training,raw_npy_folder)
    else:
        raw_training=np.load(raw_npy_folder+'raw_data.npy')
        raw_label=np.load(raw_npy_folder+'raw_labels.npy')
        normed=np.load(raw_npy_folder+'normed_raw_feature.npy')
    raw_label=np.array(raw_label)
    train_data,validation_data,test_data=seperating_validation(normed,raw_label,[0.3,0])
    if train_method=='-xgboost':
        clf=training_xgboost(train_data)
    elif train_method=='-randomforest':
        clf=training_random_forest(train_data)
        #rules=tree.export_text(clf.estimators_[0])
        #print(rules)
    print(train_data[0].shape)
    print(validation_data[0].shape)
    print(test_data[0].shape)
    predicted_train=clf.predict(train_data[0])
    predicted_val=clf.predict(validation_data[0])
    predicted_test=clf.predict(test_data[0])
    print('train : ',predicted_train.sum()/len(predicted_train),get_score(predicted_train,train_data[1]))
    print('validation : ',predicted_val.sum()/len(predicted_val),get_score(predicted_val,validation_data[1]))
    print('test : ',predicted_test.sum()/len(predicted_test),get_score(predicted_test,test_data[1]))
