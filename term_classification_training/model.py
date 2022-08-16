from transformers import AutoModel
import torch
from torch import nn


class EntityTypeClassification(nn.Module):
    def __init__(self, bert_model_path, num_class):
        super(EntityTypeClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_path)
        self.num_class = num_class
        self.classifier = nn.Linear(768, self.num_class)
        self.loss_fn = nn.CrossEntropyLoss()

    def predict(self, x, entity_place):
        h = self.bert(x)[0] # batch_size * seq_length * hidden_place
        entity_place = entity_place.float()
        entity_place = entity_place / torch.sum(entity_place, dim=1, keepdim=True) # batch_size * seq_length
        h_entity = torch.bmm(entity_place.unsqueeze(1), h).squeeze(1) # batch_size * hidden_place
        logits = self.classifier(h_entity)
        return logits

    def forward(self, x, entity_place, labels):
        logits = self.predict(x, entity_place)
        #import ipdb; ipdb.set_trace()
        loss = self.loss_fn(logits, labels)
        return loss, logits
