# -*- coding: utf-8 -*-


def addtype(score,tags):   
    #make a pred sequence
    pred = [eval(i.split('\t')[-2]) for i in score[1:]]
    sent_tokens = [i.split('\t')[0] for i in score]
    
    start_indexs = []
    end_indexs = []
    
    now_index = 0
    for i in range(1,len(sent_tokens)):
        start_indexs.append(now_index)
        end_indexs.append(now_index + len(sent_tokens[i]))
        now_index += (len(sent_tokens[i]) + 1)

    status = 0
    
    
    for tag in tags:
        begin = tag['begin']
        end = tag['end']
        for i in range(len(start_indexs)):
            if begin == start_indexs[i]:
                pred_tag = pred[i]
                for j in range(len(pred_tag)):
                    if pred_tag[j] != 'O':
                        pred_tag[j] = pred_tag[j] + '-' + tag['group']
                pred[i] = pred_tag
                status = 1
                if end_indexs[i] == end:
                    status = 0      
            elif status == 1:
                pred_tag = pred[i]
                for j in range(len(pred_tag)):
                    if pred_tag[j] != 'O':
                        pred_tag[j] = pred_tag[j] + '-' + tag['group']
                pred[i] = pred_tag
                if end_indexs[i] == end:
                    status = 0                

    type_score = [score[0]]
    
    for s in range(1,len(score)):
        parts = score[s].split('\t')
        parts[-2] = str(pred[s-1])
        type_score.append('\t'.join(parts))
    
    return type_score


scores = []
tags = []
temp = []
with open('post-results-train-score-BC5',encoding = 'utf-8') as f:
    for line in f:
        if line == '\n':
            scores.append(temp)
            temp = []
        else:
            temp.append(line[:-1])
            
print(len(scores))
            
with open('BC5-sents_match.txt',encoding = 'utf-8') as f:
    for line in f:
        tags.append(eval(line.strip()))
        
print(len(tags))
        
with open('post-results-train-score-BC5-type','w',encoding = 'utf-8') as f:
    for i in range(len(scores)):
        type_score = addtype(scores[i],tags[i])
        for ts in type_score:
            f.write(ts + '\n')
        f.write('\n')