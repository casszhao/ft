import torch.nn as nn
import torch.nn.functional as F
import torch

import time
from transformers import AdamW, get_linear_schedule_with_warmup, GPT2PreTrainedModel, GPT2Model
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
import datetime


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



class GPT2_multilabel_clf(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformers = GPT2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
    ):

        outputs = self.transformers(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 0: last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        #                       – Sequence of hidden-states at the output of the last layer of the model.
        # 1: past_key_values (optional, returned when use_cache=True is passed)
        # 2: hidden_states (optional, returned when output_hidden_states=True is passed ,
        #                   one for the output of the embeddings + one for the output of each layer)
        #                  of shape (batch_size, sequence_length, hidden_size).
        # 3: attentions

        pooled_output = outputs[0].permute(0,2,1) # (batch_size, hidden_size, sequence_length)
        pooled_output = F.max_pool1d(pooled_output, pooled_output.shape[2]).squeeze(2)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()#.to(device)
            loss = loss_fct(logits, labels)
            output = (loss, logits)
        else:
            output = logits

        return output  # (loss), logits

class GPT2_multiclass_clf(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformers = GPT2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
    ):

        outputs = self.transformers(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 0: last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        #                       – Sequence of hidden-states at the output of the last layer of the model.
        # 1: past_key_values (optional, returned when use_cache=True is passed)
        # 2: hidden_states (optional, returned when output_hidden_states=True is passed ,
        #                   one for the output of the embeddings + one for the output of each layer)
        #                  of shape (batch_size, sequence_length, hidden_size).
        # 3: attentions

        pooled_output = outputs[0].permute(0,2,1) # (batch_size, hidden_size, sequence_length)
        pooled_output = F.max_pool1d(pooled_output, pooled_output.shape[2]).squeeze(2)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()#.to(device)
            loss = loss_fct(logits, labels)
            output = (loss, logits)
        else:
            output = logits

        return output  # (loss), logits

