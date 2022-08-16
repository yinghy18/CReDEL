# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
import json

all_len = []
all_score = []
all_token = []
all_y = []

d = {2:'B',1:'I',0:'O',90:90}

def cuowei(lis):
    lis.insert(0,'O')
    lis.pop(-1)
    return lis

with open('../lensBC5train') as f:
    for line in tqdm(f):
        all_len.append(eval(line[:-1]))
        
with open('../train_BC5.json',encoding = 'utf-8') as f:
    a = json.load(f)
    
for sample in a:
    all_token.append(sample['str_words'])
        
with open('post-train-score-BC5',encoding = 'utf-8') as f:
    for line in tqdm(f):
        all_score.append(eval(line[7:-1]))
        
print('yes')
n = len(all_score)/3
assert n == int(n)

n = int(n)
        
score_B = [all_score[3*i] for i in tqdm(range(n))]
score_I = [all_score[3*i+1] for i in tqdm(range(n))]
score_O = [all_score[3*i+2] for i in tqdm(range(n))]

all_score = []

scores = torch.tensor([score_O,score_I,score_B]).permute(1,2,0)

print(scores.shape) #n,  256, 3

indice = []
scor = []

for i in range(1):
    print(i)
    a = scores[i*100000:(i+1)*100000]
    out = torch.topk(-a,1)
    indice.extend(out[1].squeeze(-1).numpy().tolist())
    scor.extend(a.numpy().tolist())
    
scores = scor

with open('post-results-train-score-BC5','a',encoding = 'utf-8') as f:
    for i in tqdm(range(n)):
        indx = 0
        f.write('[CLS]'+'\t'+'y\t'+d[indice[i][indx]]+'\t'+'score'+str(scores[i][indx:indx+1])+'\n')
        indx += 1
        for j in range(len(all_len[i])):
            towrite = all_token[i][j]+'\t'+'y\t'
            towrite = towrite +str([d[indice[i][t]] for t in range(indx,min(255,indx+all_len[i][j]))])+'\t'
            towrite = towrite + 'score'+str(scores[i][indx:indx+all_len[i][j]])+'\n'
            indx = indx + all_len[i][j]
            f.write(towrite)
            if indx > 255:
                break
            
        f.write('\n')  




        
            