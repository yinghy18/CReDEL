# -*- coding: utf-8 -*-


import torch
import numpy as np
import torch.nn as nn
from model import AverageMeter,get_std_opt,BertForTokenClassification
from transformers import BertConfig
from data import cuowei,luanxu,quanb
import time
from torch.utils.data import DataLoader,TensorDataset
import os
from copy import deepcopy
from data import align
from tqdm import tqdm
import json

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:1")

B=2
I = 1
O = 0

def noisey(y,p=0.2):
    y1 = deepcopy(y)
    for i in range(len(y)):
        a = np.random.rand(1)[0]
        if a < p:
            if y[i] == O or y[i] == 90:
                1
            elif y[i] == I:
                if i == len(y)-1:
                    y1[i] = O
                elif y[i+1] != I:
                    y1[i] = O
                else:
                    y1[i] = B
            else:
                if i == len(y)-1:
                    y1[i] = O
                elif y[i+1] != I:
                    y1[i] = O
                else:
                    y1[i] = O
                    y1[i+1] = B

    return y1


def getdata():
    text_id = []
    tag_id = []
    atmk = []
    all_len = []
    
    
    if os.path.exists('./data/XBC5train.pt') and os.path.exists('./data/YBC5train.pt') and os.path.exists('./data/attentionmaskBC5train.pt'):
        text_id = torch.load('./data/XBC5train.pt')
        tag_id = torch.load('./data/YBC5train.pt')
        atmk = torch.load('./data/attentionmaskBC5train.pt')
        
        print('text shape:',text_id.shape)
        print('tag shape:',tag_id.shape)
        print('mask shape:',atmk.shape)
        
        return text_id,tag_id,atmk
    
    
    f = open('train_BC5.json')
    xxxxx = json.load(f)
    for sample in tqdm(xxxxx):
        tx = sample['str_words']
        tg = sample['tags']
                 
        if len(tx) > len(tg):
            print(1)
            tx = tx[:len(tg)]
        elif len(tx) < len(tg):
            print(2)
            tx = tx + ['O'] * (len(tg)-len(tx))
            
        tokenized = align(tx,tg,tg,tg)
        text_id.append(tokenized[0])
        tag_id.append(tokenized[1])
        atmk.append(tokenized[4])
        all_len.append(tokenized[5])
            
    with open('lensBC5train','w') as f:
        for lens in all_len:
            f.write(str(lens)+'\n')
            
            
    text_id = torch.tensor(text_id)
    tag_id = torch.tensor(tag_id)
    atmk = torch.tensor(atmk)
    
    torch.save(text_id,'./data/XBC5train.pt')
    torch.save(tag_id,'./data/YBC5train.pt')
    torch.save(atmk,'./data/attentionmaskBC5train.pt')
    
    print('text shape:',text_id.shape)
    print('tag shape:',tag_id.shape)
    print('mask shape:',atmk.shape)
        
    
        
    return text_id,tag_id,atmk

if __name__ == '__main__':
    text_id,tag_id,atmk = getdata()
    
    test_dl = DataLoader(dataset=TensorDataset(text_id,tag_id,atmk),
    batch_size=1,
    shuffle=False,
    pin_memory=True)
    
    f = open('./results/post-train-score-BC5','a',encoding = 'utf-8')
    model = BertForTokenClassification(BertConfig(num_labels = 3))
    with torch.no_grad():
        model_dict = torch.load('model/model_checkpoint99999.pth',map_location=device)['model']
        op_dict = torch.load('model/model_checkpoint99999.pth',map_location=device)['optimizer']
        model.load_state_dict(model_dict)
        model = model.to(device)
        optimizer = get_std_opt(model)
        
        model.eval()
        
        for j,(x,Y,at_mask) in tqdm(enumerate(test_dl)):
            x = x.to(device)
            at_mask = at_mask.to(device)
            l1 = torch.tensor([2]*Y.shape[1]).long()
            l2 = torch.tensor([1]*Y.shape[1]).long()
            l3 = torch.tensor([0]*Y.shape[1]).long()
            l1 = torch.where(Y == 90, Y,l1).cuda()
            l2 = torch.where(Y == 90, Y,l2).cuda()
            l3 = torch.where(Y == 90, Y,l3).cuda()
            
            
            score1,score2,score3 = model.forward(x,attention_mask = at_mask,labels1 = l1,labels2 = l2,labels3 = l3,evaluation = True)
            score1 = score1.cpu().detach().numpy().tolist()
            score2 = score2.cpu().detach().numpy().tolist()
            score3 = score3.cpu().detach().numpy().tolist()
            f.write('score 1'+str(score1)+'\n')
            f.write('score 2'+str(score2)+'\n')
            f.write('score 2'+str(score3)+'\n')
            
            
            
            
        
