# -*- coding: utf-8 -*-

from tqdm import tqdm
import json
import time
import numpy as np

d = {'B':1,'I':2,'O':0,'A':-100,2:'B',1:'I',0:'O','2':'I','1':'B','0':'O'}
id_to_tag = json.load(open('../tag_to_id_BC5.json'))

id_to_tag_inv = {}
for k in id_to_tag.keys():
    id_to_tag_inv[str(id_to_tag[k])] = k

id_to_tag['A'] = -100
id_to_tag['0'] = 0

def newtagging(ori_tag,tag,score,types):
    if ori_tag == 'abcde':
        1
    else:
        if ori_tag == 'O':
            to_change = [ori_tag]*len(tag)
        else:
            ori_tag = id_to_tag_inv[ori_tag]
            #'B-chemical'
            to_change = [ori_tag]*len(tag)
        for i in range(len(score)):
            mins = min(score[i])
            tagpred = score[i].index(mins)
            #0,1,2 d-> O I B
            if d[tagpred] != to_change[i][0] and score[i][tagpred]< 1:
                #O change
                #B I in types change
                if d[tagpred] == 'O':
                    to_change[i] = d[tagpred]
                elif len(tag[0]) > 1 and tag[0][2:] in types:
                    to_change[i] = d[tagpred]
        a = [t[0] != ori_tag[0] for t in to_change]
        if sum(a) > len(tag)/2:
            return 'A'
        else:
            return ori_tag
            
all_len = []
all_score = []
all_token = []
all_y = []
all_res = []
all_acc = []
res = []

def f1(pred,true):
    acc = np.sum(np.array(pred) == np.array(true))/len(pred)
    return acc
    
    

with open('../train_BC5.json',encoding = 'utf-8') as f:
    di = json.load(f)
    for sample in tqdm(di):
        b = sample['tags']
        b.insert(0,'O')
        all_y.append(b)
        
print(len(all_y))
        
with open('post-results-train-score-BC5-type',encoding = 'utf-8') as f:
    for line in tqdm(f):
        if line == '\n':
            all_res.append(res)
            res = []
        else:
            res.append(line[:-1])
            
print(len(all_res))

with open('nega1-train-score-BC5our-type','w',encoding = 'utf-8') as f:
    for i in tqdm(range(len(all_res))):
        for j in range(len(all_res[i])):
            f.write('truey\t'+str(all_y[i][j])+'\t'+all_res[i][j]+'\n')
            
        f.write('\n')

with open('train-score-BC5-type',encoding = 'utf-8') as f:
    with open('train-score-BC5-type-newtag','w',encoding = 'utf-8') as g:
        for line in tqdm(f):
            if line == '\n':
                g.write(line)
                continue
            a = line[:-1].split('\t')
            if len(a)<5:
                continue
            if a[4]=='\t' or a[5][5:]=='':
                g.write(line)
                continue
            if a[4] in ['B','I','O'] or a[4]=='[]' or a[4]=='y' or len(eval(a[4]))!=len(eval(a[5][5:])):
                g.write(line)
                continue
            newtag = newtagging(a[1],eval(a[4]),eval(a[5][5:]),types = ['DISO','CHEM'])
            a[1] = newtag
            g.write('\t'.join(a)+'\n')

all_dics = []
with open('train-score-BC5-type-newtag',encoding = 'utf-8') as g:
    with open('train-score-BC5-typerefined.json','w',encoding = 'utf-8') as f:
        sample = {'str_words':[],'tags':[]}
        for line in tqdm(g):
            if line == '\n':
                all_dics.append(sample)
                sample = {'str_words':[],'tags':[]}
            else:
                a = line[:-1].split('\t')
                if len(a)<5:
                    print(line)
                    continue
                if a[4]=='y' or a[4]=='\t':
                    continue
                sample['str_words'].append(a[2])
                sample['tags'].append(id_to_tag[a[1]])
               
        f.write(json.dumps(all_dics))
                
                
            
        
        
        
        
        
    