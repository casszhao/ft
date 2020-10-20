'''apply fine-tuned bert based modle on four datasets'''

from transformers import BertPreTrainedModel, BertModel
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from transformers import AdamW
import numpy as np
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

parser = argparse.ArgumentParser(description='run fine-tuned model on multi-label dataset')

# 1
parser.add_argument('--FTModel', type=str, help= 'where is the saved trained language model, including path and name')
parser.add_argument('--BertModel', type=str, action='store', choices = ['Bert','RoBerta','XLM', 'XLNet', 'ELECTRA', 'gpt2', 'bert_xlm'])
# 2
parser.add_argument('-e', '--epochs', type=int, default=3, metavar='', help='how many epochs')
# 3
group = parser.add_mutually_exclusive_group()
group.add_argument('--running', action='store_true', help='running using the original big dataset')
group.add_argument('--testing', action='store_true', help='testing')
# 3
args = parser.parse_args()


MAX_LEN = 100
NUM_LABELS = 6
batch_size = 16
epochs = args.epochs

from multi_label_fns import validate_multilable, train_multilabel

if args.testing:
    data = pd.read_csv('./data/multi-label.csv', header=0).sample(500)
    train, test = train_test_split(data, test_size=0.2)
    test, validation = train_test_split(test, test_size=0.5)
    print(test['identity_hate'].value_counts())
elif args.running:
    data = pd.read_csv('./data/multi-label.csv', header=0)
    train, test = train_test_split(data, test_size=0.2, stratify=data['identity_hate'])
    test, validation = train_test_split(test, test_size=0.5, stratify=test['identity_hate'])
    print(test['identity_hate'].value_counts())
else:
    print('need to define parameter, it is "--running" or "--testing"')





if args.FTModel != None:
    model_name = str(args.FTModel)
elif args.BertModel != None:
    if args.BertModel == 'Bert':
        model_name = 'bert-base-cased'
    elif args.BertModel == 'RoBerta':
        model_name = 'roberta-base'
    elif args.BertModel == 'XLM':
        model_name = 'xlm-mlm-enfr-1024'
    elif args.BertModel == 'gpt2':
        model_name = 'gpt2'
else:
    print('the model name is not set up, it should be from a pretrained model file(as args.FTModel) or '
          'bert-base-cased or roberta-base or xlm-mlm-enfr-1024')
print('model_name: ', model_name)









sentences_train = train.comment_text.values
sentences_test = test.comment_text.values
sentences_validation = validation.comment_text.values

labels_train = train.iloc[:, -6:].copy()
labels_test = test.iloc[:, -6:].copy()
labels_validation = validation.iloc[:, -6:].copy()

train_labels = torch.tensor([labels_train['toxic'].values,
                             labels_train['severe_toxic'].values,
                             labels_train['obscene'].values,
                             labels_train['threat'].values,
                             labels_train['insult'].values,
                             labels_train['identity_hate'].values, ]).permute(1, 0).to(device)

test_labels = torch.tensor([labels_test['toxic'].values,
                            labels_test['severe_toxic'].values,
                            labels_test['obscene'].values,
                            labels_test['threat'].values,
                            labels_test['insult'].values,
                            labels_test['identity_hate'].values, ]).permute(1, 0).to(device)

validation_labels = torch.tensor([labels_validation['toxic'].values,
                                  labels_validation['severe_toxic'].values,
                                  labels_validation['obscene'].values,
                                  labels_validation['threat'].values,
                                  labels_validation['insult'].values,
                                  labels_validation['identity_hate'].values, ]).permute(1, 0).to(device)











if (('RoBerta' in model_name) or ('roberta' in model_name)):
    from transformers import RobertaTokenizer, RobertaModel
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False)
    from multi_label_fns import RoBerta_clf
    model = RoBerta_clf.from_pretrained(model_name,
                                        num_labels=NUM_LABELS,
                                        output_attentions=False,
                                        output_hidden_states=True)
    print('using RoBerta:', model_name)

elif (('Bert' in model_name) or ('bert' in model_name)):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    from multi_label_fns import Bert_clf
    model = Bert_clf.from_pretrained(model_name,
                                     num_labels=NUM_LABELS,
                                     output_attentions=False,
                                     output_hidden_states=True)
    print('using Bert:', model_name)

elif (('XLM' in model_name) or ('xlm' in model_name)):
    from transformers import XLMTokenizer
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-enfr-1024', do_lower_case=False)
    from multi_label_fns import XLM_clf
    model = XLM_clf.from_pretrained(model_name,
                                    num_labels=NUM_LABELS,
                                    output_attentions=False,
                                    output_hidden_states=True)
    print('using XLM:', model_name)

elif 'gpt2' in model_name:
    from transformers import GPT2Tokenizer, GPT2PreTrainedModel, GPT2DoubleHeadsModel
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
    tokenizer.cls_token = tokenizer.cls_token_id
    tokenizer.pad_token = tokenizer.eos_token
    from gpt2 import GPT2_multilabel_clf

    model = GPT2_multilabel_clf.from_pretrained(model_name,
                                     num_labels=NUM_LABELS,
                                     output_attentions=False,
                                     output_hidden_states=False,
                                     use_cache=False,
                                     )
    print(' ')
    print('using GPT2:', model_name)



model.to(device)


train_inputs = torch.Tensor()
train_masks = torch.Tensor()
for sent in sentences_train:
    encoded_sent = tokenizer.encode_plus(sent,  # Sentence to encode.
                                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                         max_length=MAX_LEN,  # Truncate all sentences.
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         return_token_type_ids=False,
                                         truncation=True,
                                         return_tensors='pt')  # return pytorch not tensorflow tensor
    train_inputs = torch.cat((train_inputs, encoded_sent['input_ids'].float()), dim=0)
    train_masks = torch.cat((train_masks, encoded_sent['attention_mask'].float()), dim=0)
train_inputs.to(device)
train_masks.to(device)

validation_inputs = torch.Tensor()
validation_masks = torch.Tensor()
for sent in sentences_validation:
    encoded_sent = tokenizer.encode_plus(sent,  # Sentence to encode.
                                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                         max_length=MAX_LEN,  # Truncate all sentences.
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         return_token_type_ids=False,
                                         truncation=True,
                                         return_tensors='pt')  # return pytorch not tensorflow tensor
    validation_inputs = torch.cat((validation_inputs, encoded_sent['input_ids'].float()), dim=0)
    validation_masks = torch.cat((validation_masks, encoded_sent['attention_mask'].float()), dim=0)
validation_inputs.to(device)
validation_masks.to(device)

test_inputs = torch.Tensor()
test_masks = torch.Tensor()
for sent in sentences_test:
    encoded_sent = tokenizer.encode_plus(sent,  # Sentence to encode.
                                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                         max_length=MAX_LEN,  # Truncate all sentences.
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         return_token_type_ids=False,
                                         truncation=True,
                                         return_tensors='pt')  # return pytorch not tensorflow tensor
    test_inputs = torch.cat((test_inputs, encoded_sent['input_ids'].float()), dim=0)
    test_masks = torch.cat((test_masks, encoded_sent['attention_mask'].float()), dim=0)

test_inputs.to(device)
test_masks.to(device)

# for training data
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# for validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)



def metrics(rounded_preds, label):
    """
    preds (batch size, 6) before sigmoid
    label (batch size, 6)
    """
    #rounded_preds = torch.round(torch.sigmoid(preds))  # (batch size, 6)
    pred_array = rounded_preds.cpu().detach().numpy()
    label_array = label.cpu().detach().numpy()

    #correct = (pred_array == label).float()  # convert into float for division
    #acc = correct.sum() / len(correct)

    micro_f1 = f1_score(label_array, pred_array, average='micro', zero_division=1)
    macro_f1 = f1_score(label_array, pred_array, average='macro', zero_division=1)
    return micro_f1, macro_f1

'''
================== Training Loop =======================
'''
optimizer = AdamW(model.parameters(), lr=0.0005, weight_decay = 0.01, eps = 1e-6)
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps= int(total_steps*0.06),  # Default value in run_glue.py
                                            num_training_steps=total_steps)


best_valid_loss = float('inf')
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    print("")
    print('========== Epoch {:} / {:} =========='.format(epoch_i + 1, epochs))
    t0 = time.time()

    train_loss = train_multilabel(model, train_dataloader)

    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Running Validation...")

    t0 = time.time()
    valid_loss = validate_multilable(model, validation_dataloader)
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

#torch.save(model.state_dict(), str(args.resultpath) + resultname + '_model.pt')

print("")
print("Training complete!")

prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, shuffle = False)

model.eval()

predictions = torch.Tensor().to(device)



labels = torch.Tensor().to(device)
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        # Forward pass, calculate logit predictions, 没有给label, 所以不outputloss
        outputs = model(b_input_ids.long(), token_type_ids=None,
                        attention_mask=b_input_mask)  # return: loss(only if label is given), logit
    #logits = outputs
    rounded_preds = torch.round(torch.sigmoid(outputs))
    predictions = torch.cat((predictions, rounded_preds))  #rounded_preds.float()
    labels = torch.cat((labels, b_labels.float()))
print(' prediction    DONE.')

pred_array = predictions.cpu().detach().numpy()
label_array = labels.cpu().detach().numpy()


micro_f1 = f1_score(label_array, pred_array, average='micro', zero_division=1)
macro_f1 = f1_score(label_array, pred_array, average='macro', zero_division=1)

print("micro is {}, macro is {}".format(micro_f1, macro_f1))

#predictions_np = predictions.cpu().numpy()
predictions_df = pd.DataFrame(pred_array,
                              columns = ['pred_toxic', 'pred_severe_toxic', 'pred_obscene', 'pred_threat', 'pred_insult', 'pred_identity_hate'])
print(' =========== predictions_df ============')
print(predictions_df)


result = test.join(predictions_df)

f1_toxic = f1_score(result['toxic'], result['pred_toxic'], zero_division =1 )
f1_severe_toxic = f1_score(result['severe_toxic'], result['pred_severe_toxic'], zero_division =1)
f1_obscene = f1_score(result['obscene'], result['pred_obscene'], zero_division =1)
f1_threat = f1_score(result['threat'], result['pred_threat'], zero_division =1)
f1_insult = f1_score(result['insult'], result['pred_insult'], zero_division =1)
f1_identity_hate = f1_score(result['identity_hate'], result['pred_identity_hate'], zero_division =1)
print("f1_toxic:", f1_toxic)
print("f1_severe_toxic:", f1_severe_toxic)
print("f1_threat:", f1_threat)
print("f1_obscene:", f1_obscene)
print("f1_insult:", f1_insult)
print("f1_identity_hate:", f1_identity_hate)
print("macro F1:", (f1_toxic + f1_severe_toxic + f1_obscene + f1_threat + f1_insult + f1_identity_hate)/6)

