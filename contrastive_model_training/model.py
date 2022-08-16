import torch
import numpy as np
from transformers import  BertPreTrainedModel, BertTokenizer, BertModel
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class CEvalue(nn.Module):
    def __init__(self,reduction='mean',eps = 0.2,ignore_index = 90):
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self,logits,labels):
        #logits:(N,C),labels:(N)
        loss = torch.zeros(len(labels))
        index = []
        for i in range(len(labels)):
            if labels[i] != self.ignore_index:
                loss[i] = torch.exp(logits[i][labels[i]])/torch.sum(torch.exp(logits[i]))
                index.append(i)
            
        loss = -torch.log((loss + self.eps)/( 1 + self.eps))
        if self.reduction == 'mean':
            return torch.mean(loss[index])
        else:
            return loss
        

class labelscore:
    
    def __init__(self,loss_fct,num_labels,attention_mask=None):
        self.loss_fct = loss_fct
        self.num_labels = num_labels
        self.attention_mask = attention_mask
        
    # Only keep active parts of the loss
    def loss(self,logits,labels):
        if self.attention_mask is not None:
            active_loss = self.attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(90).type_as(labels)
                    )
            loss = self.loss_fct(active_logits, active_labels)
        else:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
          
        return loss

class BertForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False).cuda()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_id=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels1=None,
        labels2=None,
        labels3=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        evaluation = False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output = self.bert(
            input_id,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        #1 should be better than 2

        sequence_output = output[0]


        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if not evaluation:
            CE = CEvalue()
            score_fct = CE.forward
            labelscorefct = labelscore(score_fct,self.num_labels,attention_mask)
        
            loss_fct = torch.nn.TripletMarginWithDistanceLoss(distance_function=labelscorefct.loss,margin=0.4)
            loss = loss_fct(logits,labels1,labels2)
        
        if evaluation:
            CE = CEvalue(reduction = 'none')
            score_fct = CE.forward
            labelscorefct = labelscore(score_fct,self.num_labels,attention_mask)
            
            if labels3 == None:
                return (labelscorefct.loss(logits,labels1),labelscorefct.loss(logits,labels2))
            else:
                return (labelscorefct.loss(logits,labels1),labelscorefct.loss(logits,labels2),labelscorefct.loss(logits,labels3))


        return (loss,logits)



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 2*self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(768, 2, 1000,
            torch.optim.Adam(model.parameters(), lr=8e-6, betas=(0.9, 0.98), eps=1e-9))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        



