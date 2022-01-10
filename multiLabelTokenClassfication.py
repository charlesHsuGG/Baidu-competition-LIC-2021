
# -*- coding: utf-8 -*-
"""This is an CRF Modules based on pytorch implement."""
__author__ = "Charles_Hsu"

from typing import Optional

import torch
import torch.nn as nn

from transformers import AutoModel

from partial_crf import PartialCRF

class MultiLabelTokenClassification(nn.Module):
    
    def __init__(self, model_name, config):
        super().__init__()
        
        self.num_labels = config.num_labels
        self.word_embedding = AutoModel.from_pretrained(model_name, config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(
        self, input_ids, attention_mask = None, token_type_ids = None, labels = None
    ):
        outputs = self.word_embedding(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        all_encoder_layers = outputs.last_hidden_state
        
        outputs = self.dropout(all_encoder_layers)
        logits = self.classifier(outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            
        return loss, logits
    
    
class MultiLabelTokenClassificationCRF(nn.Module):
    
    def __init__(self, model_name, config):
        super().__init__()
        
        self.num_labels = config.num_labels
        self.word_embedding = AutoModel.from_pretrained(model_name, config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = PartialCRF(self.num_labels)
        
    def forward(
        self, input_ids, attention_mask = None, token_type_ids = None, labels = None
    ):
        outputs = self.word_embedding(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        all_encoder_layers = outputs.last_hidden_state
        
        outputs = self.dropout(all_encoder_layers)
        logits = self.classifier(outputs)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        logits = self.loss_fct.marginal_probabilities(logits).transpose(0, 1)
        return loss, logits