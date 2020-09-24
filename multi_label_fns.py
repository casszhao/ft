from transformers import BertPreTrainedModel, BertModel, BertConfig, RobertaConfig, \
    XLMConfig, XLMModel, XLMPreTrainedModel, RobertaModel, GPT2PreTrainedModel, GPT2Model
import torch.nn as nn
import torch.nn.functional as F
import torch

import time
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


class BertForMTL(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
        output_hidden_states=None,
        return_dict=None,
    ):

        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.NLLLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class Bert_clf(BertPreTrainedModel):
    def __init__(self, config, token='cls'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.token = token

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
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 0: last_hidden_state
        # 1: pooler_output
        # 2: hidden_states (one for the output of the embeddings + one for the output of each layer)
        #                  of shape (batch_size, sequence_length, hidden_size).
        # 3: attentions
        if self.token == 'embedding':
            hidden_states = outputs[2]
            output_of_each_layer = hidden_states[0]
            output_oel = self.dropout(output_of_each_layer).permute(0, 2, 1)
            # [16, 100, 256] permute --> [16, 256, 100]
            pooled = F.max_pool1d(output_oel, output_oel.shape[2]).squeeze(2)
            # [16, 256]
            logits = self.classifier(pooled)
        elif self.token == 'cls':
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            print('need to define using [CLS] token or embedding to the nn.linear layer')

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()#.to(device)
            loss = loss_fct(logits, labels)
            output = (loss, logits)
        else:
            output = logits

        return output  # (loss), logits


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RoBerta_clf(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
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
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()#.to(device)
            loss = loss_fct(logits, labels)
            output = (loss, logits)
        else:
            output = logits

        return output

class GPT2_clf(GPT2PreTrainedModel):
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


class XLM_clf(XLMPreTrainedModel):
    def __init__(self, config, token = 'cls'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLMModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.token = token
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 0. last_hidden_state size: [batch_size, sequence_length, hidden_size]
        # 1. hidden_states
        #    (output of the embeddings + output of each layer) size: [batch_size, sequence_length, hidden_size]
        # 2. attentions

        if self.token == 'embedding':
            hidden_states = outputs[1]
            output_of_each_layer = hidden_states[0]
            output_oel = self.dropout(output_of_each_layer).permute(0, 2, 1)
            # [16, 100, 256] permute --> [16, 256, 100]
            pooled = F.max_pool1d(output_oel, output_oel.shape[2]).squeeze(2)
            # [16, 256]
            logits = self.classifier(pooled)
        elif self.token == 'cls':
            pooled_output = outputs[0]
            pooled_output = self.dropout(pooled_output).permute(0, 2, 1)
            # [batch_size, sequence_length, hidden_size] -> [batch_size, hidden_size, sequence_length]
            pooled = F.max_pool1d(pooled_output, pooled_output.shape[2]).squeeze(2)
            logits = self.classifier(pooled)
        else:
            print('need to define using [CLS] token or embedding to the nn.linear layer')

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()#.to(device)
            loss = loss_fct(logits, labels)
            output = (loss, logits)
        else:
            output = logits

        return output


class Bert_hidden(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 0: last_hidden_state
        # 1: pooler_output
        # 2: hidden_states (one for the output of the embeddings + one for the output of each layer)
        #                  of shape (batch_size, sequence_length, hidden_size).
        # 3: attentions

        return outputs[0]  # (loss), logits


class XLM_hidden(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLMModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 0. last_hidden_state size: [batch_size, sequence_length, hidden_size]
        # 1. hidden_states
        #    (output of the embeddings + output of each layer) size: [batch_size, sequence_length, hidden_size]
        # 2. attentions

        return outputs[0]


class Ensemble_clf(BertPreTrainedModel, XLMPreTrainedModel):
    def __init__(self, config):
        BertPreTrainedModel.__init__(config)
        XLMPreTrainedModel.__init__(config)
        self.num_labels = BertPreTrainedModel.config.num_labels
        self.bert = BertModel(config)

        self.classifier = nn.Linear(BertPreTrainedModel.config.hidden_size + XLMPreTrainedModel.config.hidden_size, config.num_labels)
        self.init_weights()

        #self.num_labels = config.num_labels
        self.transformer = XLMModel(config)
        self.init_weights()

        self.dropout = nn.Dropout(0.1)

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
            langs=None,
    ):

        xlm_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
            # 0. last_hidden_state size: [batch_size, sequence_length, hidden_size]
            # 1. hidden_states
            #    (output of the embeddings + output of each layer) size: [batch_size, sequence_length, hidden_size]
            # 2. attentions

        xlm_last_hidden_state = outputs[0]


        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # 0: last_hidden_state
        # 1: pooler_output
        # 2: hidden_states (one for the output of the embeddings + one for the output of each layer)
        #                  of shape (batch_size, sequence_length, hidden_size).
        # 3: attentions

        bert_last_hidden_state = bert_outputs[0] #[batch_size, sequence_length, hidden_size]

        cat_hidden_state = torch.cat((bert_last_hidden_state, xlm_last_hidden_state), dim = 1) #[batch_size, sequence_length, hidden_size_1 + hiddent_size_2]

        cat_hidden_state = self.dropout(cat_hidden_state).permute(0, 2, 1)
            # [16, 100, hidden 1 + hidden 2] permute --> [16, hidden 1 + hidden 2, 100]
        pooled = F.max_pool1d(cat_hidden_state, cat_hidden_state.shape[2]).squeeze(2)
            # [16, 256]
        logits = self.classifier(pooled)


        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()#.to(device)
            loss = loss_fct(logits, labels)
            output = (loss, logits)
        else:
            output = logits

        return output  # (loss), logits


### functions not models

def validate_multilable(model, dataloader):
    print(" === Validation ===")
    model.eval()
    valid_loss, f1_micro_total = 0, 0

    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids = batch[0].long()
        b_input_mask = batch[1].long()
        b_labels = batch[2].float()

        with torch.no_grad():
            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        rounded_preds = torch.round(torch.sigmoid(logits))  # (batch size, 6)
        prediction = rounded_preds.detach().cpu().numpy()

        labels = b_labels.to('cpu').numpy()
        f1_micro = f1_score(labels, prediction, average='micro', zero_division=1)
        f1_micro_total += f1_micro

        valid_loss += loss

    return valid_loss / len(dataloader), f1_micro_total / len(dataloader)
    # Report the final accuracy for this validation run.


epochs = 3
def train_multilabel(model, dataloader):
    from transformers import get_linear_schedule_with_warmup
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    model.train()
    total_loss = 0

    for step, batch in enumerate(dataloader):

        if step % 2000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))

        b_input_ids = batch[0].long().to(device)
        b_input_mask = batch[1].long().to(device)
        b_labels = batch[2].float().to(device)

        optimizer.zero_grad()

        loss, logit = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            )

        total_loss += loss.item()

        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    train_loss_this_epoch = total_loss / len(dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(train_loss_this_epoch))
    return train_loss_this_epoch
