# -*- coding: utf-8 -*-

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer,BertTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import os
from copy import deepcopy
import json
from tqdm import tqdm

def cuowei(tensor):
    b = tensor.numpy().tolist()
    b.insert(0,0)
    b.pop(-1)
    
    b = torch.tensor(b).long()
    b = torch.where(tensor == 90, tensor,b)
    return b

def quanb(tensor):
    b = torch.ones(len(tensor))*2
    b = b.long()
    b = torch.where(tensor == 90, tensor,b)
    
    return b

def luanxu(tensor):
    n = torch.sum(tensor!=90).item()
    x = torch.randperm(n)
    tensor[:n] = tensor[:n][x]
    
    return tensor
    
    

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
B = 'B'
I = 'I'
O = 'O'
d = {B:2,I:1,O:0,90:90}

def makedatay():
    y = []
    with open('tagging',encoding = 'utf-8') as f:
        for line in f:
            y.append(line[:-1].split(' '))
    y1_all = []
    y2_all = []
    for y_all in y:        
        numofen = y_all.count(B)
        if numofen == 0:
            y1 = y_all
            y1_all.append(y1)
            y2 = noisey(y1)
            
            y2_all.append(y2)
            continue
    
            
        delindex = np.random.randint(numofen)
        count = -1
        y1 = []
        for i in range(len(y_all)):
            if y_all[i] == B:
                count += 1
            if count == delindex:
                y1.append(O)
            else:
                y1.append(y_all[i])
                
        y1_all.append(y1)
        y2 = noisey(y1)
            
        y2_all.append(y2)
    
    return y,y1_all,y2_all

def pad(t,maxlen=256):
    #do pad and truncate
    a = len(t)
    if a < maxlen:
        t = t + [90]*(maxlen-a)
        at_mk = [1]*a + [0]*(maxlen-a)
        if len(t)!=maxlen or len(at_mk)!=maxlen:
            print(t)
            print(at_mk)
            assert len(t)==maxlen
            assert len(at_mk = maxlen)
        return t,at_mk
    elif a == maxlen:
        at_mk = [1]*a
        if len(t)!=maxlen or len(at_mk)!=maxlen:
            print(t)
            print(at_mk)
            assert len(t)==maxlen
            assert len(at_mk = maxlen)
        return t,at_mk
    else:
        at_mk = [1]*maxlen
        t = t[:maxlen-1]+[t[-1]]
        if len(t)!=maxlen or len(at_mk)!=maxlen:
            print(t)
            print(at_mk)
            assert len(t)==maxlen
            assert len(at_mk = maxlen)
        return t,at_mk
        
        
    

def align(tx,tg,tg1,tg2):
    tx_ed = [101]
    tg_ed = [0]
    tg_ed1 = [0]
    tg_ed2 = [0]
    lens = []
    for i in range(len(tx)):
        word = tx[i]
        input_ids = torch.tensor([tokenizer.encode(word, add_special_tokens=False)]).numpy().tolist()[0]
        l = len(input_ids)
        input_tags2 = [tg2[i]]*l
        input_tags1 = [tg1[i]]*l
        input_tags = [tg[i]]*l
        tx_ed.extend(input_ids)
        tg_ed.extend(input_tags)
        tg_ed1.extend(input_tags1)
        tg_ed2.extend(input_tags2)
        lens.append(l)

    tx_ed.append(102)
    tg_ed.append(0)
    tg_ed1.append(0)
    tg_ed2.append(0)
    
    tx_ed,atmk = pad(tx_ed)
    tg_ed,_ = pad(tg_ed)
    tg_ed1,_ = pad(tg_ed1)
    tg_ed2,_ = pad(tg_ed2)

    
    return (tx_ed,tg_ed,tg_ed1,tg_ed2,atmk,lens)

def getdata():
    text_id = []
    tag1_id = []
    tag2_id = []
    tag_id = []
    atmk = []
    all_len = []
    
    if os.path.exists('./data/X.pt') and os.path.exists('./data/Y.pt') and os.path.exists('./data/Y1.pt') and os.path.exists('./data/Y2.pt') and os.path.exists('./data/attentionmask.pt'):
        text_id = torch.load('./data/X.pt')
        tag_id = torch.load('./data/Y.pt')
        tag1_id = torch.load('./data/Y1.pt')
        tag2_id = torch.load('./data/Y2.pt')
        atmk = torch.load('./data/attentionmask.pt')
        
        print('text shape:',text_id.shape)
        print('tag shape:',tag_id.shape)
        print('tag1 shape:',tag1_id.shape)
        print('tag2 shape:',tag2_id.shape)
        print('mask shape:',atmk.shape)
        
        return text_id,tag_id,tag1_id,tag2_id,atmk
    
    
    with open('train.json') as f:
        traindic = json.load(f)
        for sample in tqdm(traindic):
            tx = sample['tokens']
            tg2 = sample['y2']
            tg1 = sample['y1']
            tg = sample['y']
            
            tg2 = [d[i] for i in tg2]
            tg1 = [d[i] for i in tg1]
            tg = [d[i] for i in tg]
            
            tokenized = align(tx,tg,tg1,tg2)
            text_id.append(tokenized[0])
            tag_id.append(tokenized[1])
            tag1_id.append(tokenized[2])
            tag2_id.append(tokenized[3])
            atmk.append(tokenized[4])
            all_len.append(tokenized[5])
            
    with open('lens','w') as f:
        for lens in all_len:
            f.write(str(lens)+'\n')
            
            
    text_id = torch.tensor(text_id)
    tag_id = torch.tensor(tag_id)
    tag1_id = torch.tensor(tag1_id)
    tag2_id = torch.tensor(tag2_id)
    atmk = torch.tensor(atmk)
    
    torch.save(text_id,'./data/X.pt')
    torch.save(tag_id,'./data/Y.pt')
    torch.save(tag1_id,'./data/Y1.pt')
    torch.save(tag2_id,'./data/Y2.pt')
    torch.save(atmk,'./data/attentionmask.pt')
    
    print('text shape:',text_id.shape)
    print('tag shape:',tag_id.shape)
    print('tag1 shape:',tag1_id.shape)
    print('tag2 shape:',tag2_id.shape)
    print('mask shape:',atmk.shape)
        
    
        
    return text_id,tag_id,tag1_id,tag2_id,atmk
    
    
def noisey(y,p=0.2,seed = 0):
    np.random.seed(seed)
    for i in range(len(y)):
        if np.random.rand(1)[0] < p:
            if y[i] == 'I':
                if i == len(y)-1:
                    y[i] = 'O'
                elif y[i+1] != 'I':
                    y[i] = 'O'
                else:
                    y[i] = 'B'
            elif y[i] == 'B':
                if i == len(y)-1:
                    y[i] = 'O'
                elif y[i+1] != 'I':
                    y[i] = 'O'
                else:
                    y[i] = 'O'
                    y[i+1] = 'B'

    return y

def makedata(batch_size):
    text_id,tag_id,tag1_id,tag2_id,attn_mask = getdata()
    
    tag1_ideval = deepcopy(tag1_id)[:10000]
    tag2_ideval  = deepcopy(tag2_id)[:10000]
    text_ideval = deepcopy(text_id)[:10000]
    attn_maskeval = deepcopy(attn_mask)[:10000]
      
    num = tag1_id.shape[0]
    eval_num = tag1_ideval.shape[0]
    
    t =deepcopy(tag1_id[int(-num/5):])
    for i in range(len(t)):
        t[i] = cuowei(t[i])
        t[i] = cuowei(t[i])
    tag1_id[int(-num/5):] = deepcopy(tag2_id[int(-num/5):])
    tag2_id[int(-num/5):] = deepcopy(t)
    
    if os.path.exists('Y2eval.pt') and os.path.exists('Y1eval.pt'):
        tag2_ideval = torch.load('Y2eval.pt')
        tag1_ideval = torch.load('Y1eval.pt')
        
    else:
        for i in tqdm(range(tag1_ideval.shape[0])):
            if i <= int(eval_num/5):
                tag1_ideval[i] = cuowei(tag1_ideval[i])
            elif i <= int(eval_num/5*1.5):
                tag1_ideval[i] = quanb(tag1_ideval[i])
            elif i <= int(eval_num/5*2.5):
                tag1_ideval[i] = luanxu(tag1_ideval[i])
        
        for i in tqdm(range(tag1_ideval.shape[0])):
            if i <= int(eval_num/5):
                tag2_ideval[i] = cuowei(tag2_ideval[i])
            elif i <= int(eval_num/5*1.5):
                tag2_ideval[i] = quanb(tag2_ideval[i])
            elif i <= int(eval_num/5*2.5):
                tag2_ideval[i] = luanxu(tag2_ideval[i])

        torch.save(tag2_ideval,'Y2eval.pt')
        torch.save(tag1_ideval,'Y1eval.pt')
        
    print('datayes!')

    #THE TRUE AND THE NOISY
    
    print('eval',text_ideval.shape,tag2_ideval.shape,tag1_ideval.shape)

    train_dl2 = DataLoader(
    dataset=TensorDataset(text_ideval,tag2_ideval,tag1_id[:10000],attn_maskeval),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    )

    valid_dl = DataLoader(
    dataset=TensorDataset(text_ideval,tag1_ideval,tag_id[:10000],attn_maskeval),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    )
    
    if os.path.exists('Y2train.pt') and os.path.exists('Y1train.pt'):
        tag2_idtrain = torch.load('Y2train.pt')
        tag1_idtrain = torch.load('Y1train.pt')  
    else:    
        tag2_idtrain = torch.zeros(attn_mask.shape[0]*3,attn_mask.shape[1]).long()
        tag1_idtrain = deepcopy(tag1_id).repeat(3,1)
        tag1_idtrain.repeat(3,1)
        
        print(tag2_idtrain.shape)
        print(tag1_idtrain.shape)
        
    
        tag1_idtrain[:attn_mask.shape[0],] = deepcopy(tag2_id)
        for i in tqdm(range(attn_mask.shape[0])):
            tag2_idtrain[i] = cuowei(cuowei(tag1_id[i]))
            
        for i in tqdm(range(tag_id.shape[0])):
            if i <= int(num/10):
                tag2_idtrain[i+tag_id.shape[0]] = cuowei(tag2_id[i])
            elif i <= int(num/10*1.5):
                tag2_idtrain[i+tag_id.shape[0]] = quanb(tag2_id[i])
            elif i <= int(num/10*2.5):
                tag2_idtrain[i+tag_id.shape[0]] = luanxu(tag2_id[i])
            
        for i in tqdm(range(tag_id.shape[0])):
            if i <= int(num/10):
                tag2_idtrain[i+2*tag_id.shape[0]] = quanb(tag2_id[i])
            elif i <= int(num/10*2.5) and i > int(num/10*2):
                tag2_idtrain[i+2*tag_id.shape[0]] = luanxu(tag2_id[i])
            elif i > int(num/10*3) and i <= int(num/10*3.5):
                tag2_idtrain[i+2*tag_id.shape[0]] = cuowei(tag2_id[i])
    
        torch.save(tag2_idtrain,'Y2train.pt')
        torch.save(tag1_idtrain,'Y1train.pt')
        
    attn_mask = attn_mask.repeat(3,1)
    text_id = text_id.repeat(3,1)
    
    
    
    train_dl = DataLoader(
    dataset=TensorDataset(text_id,tag2_idtrain,tag1_idtrain,attn_mask),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    )
    
    print('traindata,yes!')
    
    return train_dl,train_dl2,valid_dl


    
                    
        