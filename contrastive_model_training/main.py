# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from model import AverageMeter,get_std_opt,BertForTokenClassification
from torch.utils.data import DataLoader,TensorDataset
from data import makedata,cuowei
import time
import os
from transformers import BertConfig
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# Hyper-params
max_epochs = 15
batch_size = 8
d ={2:'B',1:'I',0:'O'} # logits [batch,3] [-15,1,15] loss(xe) (e-15 e,e15)


#data and model all tensor
#y1 is 0,1,2 y1bert is bert id. Y2 is bert id.
train_dl,train_dl_eval,valid_dl = makedata(batch_size)


model = BertForTokenClassification(BertConfig(num_labels = 3)).cuda()

optimizer = get_std_opt(model)

#train!
for epoch in range(max_epochs):
    t0 = time.time()
    model.train()
    meter = AverageMeter()
    for i, (x,Y2,Y1,at_mask) in enumerate(train_dl):
        for j in range(int(Y2.shape[0]/2)):
            Y2[j] = cuowei(Y2[j])
        
        x = x.cuda()
        Y2 = Y2.cuda()
        Y1 = Y1.cuda()
        at_mask = at_mask.cuda()
        optimizer.optimizer.zero_grad()
        #labels 1 should be better than labels2
        loss,output = model.forward(x,attention_mask = at_mask,labels1 = Y1,labels2 = Y2)
        loss.backward()
        optimizer.optimizer.step()

        meter.update(loss.detach().cpu().numpy())
        if i == 100000:
            break
        
        if i%10 == 0:
            print('loss',loss.detach().cpu().numpy(),'step',i)
            
        if i%2000 == 1999:
            delta = time.time()-t0
            print('\r step {:5d} - loss {:.6f} - {:4.6f} sec per epoch'.format(i, meter.avg, delta))
    

            torch.save({
            "optimizer": optimizer.optimizer.state_dict(),
            "model": model.state_dict()
            }, './model/model_checkpoint'+str(i)+'.pth'
            )
            pred = []
            model.eval()
            f = open('./results/scoreepoch'+str(i),'a')
            g = open('./results/scoretrainepoch'+str(i),'a')
            with torch.no_grad():
                for j,(x,Y1,Y,at_mask) in tqdm(enumerate(valid_dl)):
                    x = x.cuda()
                    Y1 = Y1.cuda()
                    Y = Y.cuda()
                    at_mask = at_mask.cuda()
                    score1,score2 = model.forward(x,attention_mask = at_mask,labels1 = Y,labels2 = Y1,evaluation = True)
                    score1 = score1.cpu().detach().numpy().tolist()
                    score2 = score2.cpu().detach().numpy().tolist()
                    f.write('score 1'+str(score1)+'\n')
                    f.write('score 2'+str(score2)+'\n')
                
            
                for j,(x,Y2,Y1,at_mask) in tqdm(enumerate(train_dl_eval)):
                    x = x.cuda()
                    Y1 = Y1.cuda()
                    Y2 = Y2.cuda()
                    at_mask = at_mask.cuda()
                    score1,score2 = model.forward(x,attention_mask = at_mask,labels1 = Y1,labels2 = Y2,evaluation = True)
                    score1 = score1.cpu().detach().numpy().tolist()
                    score2 = score2.cpu().detach().numpy().tolist()
                    g.write('score 1'+str(score1)+'\n')
                    g.write('score 2'+str(score2)+'\n')
                    



