# -*- coding: utf-8 -*-
"""This is an CRF Modules based on pytorch implement."""
__author__ = "Charles_Hsu"

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
    
    
class MultiLabelStackedClassification(nn.Module):
    
    def __init__(self, models, num_labels, dropout_rate=0.2):
        super().__init__()
        self.stacked_models = nn.ModuleList(models)
        for stacked_model in self.stacked_models:
            for param in stacked_model.parameters():
                param.requires_grad = False
                
#         self.dropout = nn.Dropout(dropout_rate)
#         self.classifier = nn.Linear(len(self.stacked_models), 1)
        self.classifier = nn.Linear(len(self.stacked_models) * num_labels, num_labels)
        
    def forward(
        self, input_ids, attention_mask = None, token_type_ids = None, labels = None
    ):
        model_input_ids = input_ids.transpose(0, 1)
        model_attention_mask = attention_mask.transpose(0, 1)
        model_token_type_ids = token_type_ids.transpose(0, 1)
        output_logits = []
        for model_index, stacked_model in enumerate(self.stacked_models):
#             for param in stacked_model.parameters():
#                 print(param.get_device())
#             print(model_input_ids[model_index].get_device())
            _, logits = stacked_model(
                input_ids=model_input_ids[model_index],
                token_type_ids=model_token_type_ids[model_index],
                attention_mask=model_attention_mask[model_index]
            )
            output_logits.append(logits)
        stacked_output = torch.cat(output_logits, dim=-1)
#         stacked_output = self.dropout(stacked_output)
        logits = self.classifier(F.relu(stacked_output))
#         stacked_output = torch.stack(output_logits, dim=-1)

#         outputs = self.dropout(stacked_output)
#         logits = torch.squeeze(self.classifier(outputs), dim=-1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            
        return loss, logits