# -*- coding: utf-8 -*-


def multitag_to_one(tags):
    O = 0
    I = 0
    B = 0
    for i in tags:
        if i== 'O':
            O+=1
        elif i == 'I':
            I+=1
        else:
            B+=1
    if O==I and I == B:
        if tags != []:
            return tags[0]
        else:
            return 'O'
    else:
        vote = [O,I,B]
        tagpred = vote.index(max(vote))
        if tagpred == 0:
            return 'O'
        elif tagpred == 1:
            return 'I'
        else:
            return 'B'

def extract_term(tokens,tag):        
    terms_IO = []
    temp_IO = []
    
    for i in range(len(tag)):
        if tag[i] == 'B':
            if temp_IO != []:
                terms_IO.append(temp_IO)
                temp_IO = []
            temp_IO.append(i)
        elif tag[i] == 'I':
            temp_IO.append(i)
        elif tag[i] == 'O':
            if temp_IO != []:
                terms_IO.append(temp_IO)
                temp_IO = []
    
    if temp_IO != []:
        terms_IO.append(temp_IO)  
        
    temp_sent = ' '.join(tokens[1:])
        
    term_dict = []
    if terms_IO != []:
        print(tokens[terms_IO[0][0]:terms_IO[0][-1]+1])
    terms_IO = [k for k in terms_IO if 0 not in k]
    for term in terms_IO:
        now_index = 0
        for i in range(1,len(tokens)):
            if i == term[0]:
                begin = now_index
            if i == term[-1]:
                end = now_index + len(tokens[i])
            now_index += (len(tokens[i]) + 1)
            
        #print(temp_sent[begin:end])
        term_dict.append({'begin':begin,'end':end,'phrase':temp_sent[begin:end]})

        
    return term_dict

def tes(score):   
    #make a pred sequence
    pred = [['O']]+[eval(i.split('\t')[-2]) for i in score[1:]]
    pred = [multitag_to_one(k) for k in pred]
    sent = [i.split('\t')[0] for i in score]
    
    
    terms_IO_true = extract_term(sent,pred)                
    
    return terms_IO_true,' '.join(sent[1:])


scores = []
temp = []
with open('post-results-train-score-BC5',encoding = 'utf-8') as f:
    for line in f:
        if line == '\n':
            scores.append(temp)
            temp = []
        else:
            temp.append(line[:-1])
            
print(len(scores))
print(scores[0])

terms,sents = [],[]
for sc in scores:
    t,s = tes(sc)
    terms.append(t)
    sents.append(s)
    
with open('BC5-sents','w') as f:
    for s in sents:
        f.write(s+'\n')
        
with open('BC5-tags','w') as f:
    for t in terms:
        f.write(str(t)+'\n')
        

        
    
    




