'''apply fine-tuned bert based modle on four datasets'''

from transformers import BertPreTrainedModel, BertModel
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
# 0
parser.add_argument('--data', type=str) #, choices=['multi-label', 'wassem', 'AG10K', 'tweet50k']
# 1
parser.add_argument('--BertModel', type=str, action='store', choices = ['Bert','RoBerta','XLM', 'XLNet', 'ELECTRA', 'gpt2', 'bert_xlm'])
# 2
parser.add_argument('-e', '--epochs', type=int, default=3, metavar='', help='how many epochs')
# 3
group = parser.add_mutually_exclusive_group()
group.add_argument('--running', action='store_true', help='running using the original big dataset')
group.add_argument('--testing', action='store_true', help='testing')
# 3
# 4

parser.add_argument('--resultpath', type=str, help='where to save the result csv')
args = parser.parse_args()


MAX_LEN = 100

if args.data == 'AG10K':
    NUM_LABELS = 3
else:
    NUM_LABELS = 4

batch_size = 16
epochs = args.epochs


train_path = str(args.data) + '_train.csv'
test_path = str(args.data) + '_test.csv'
validation_path = str(args.data) + '_validation.csv'

if args.testing:
    train = pd.read_csv(train_path).sample(20)
    test = pd.read_csv(test_path).sample(20).reset_index()
    validation = pd.read_csv(validation_path).dropna()
elif args.running:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path).reset_index()
    validation = pd.read_csv(validation_path).dropna()
else:
    print('need to define parameter, it is "--running" or "--testing"')


sentences_train = train.comment.values
labels_train = train.label.values

sentences_test = test.comment.values
labels_test = test.label.values

sentences_validation = validation.comment.values
labels_validation = validation.label.values



# AG10K and tweet50k need to convert their labels to numbers
if args.data == 'AG10K':
    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()
    le.fit(["NAG", "CAG", "OAG"])

    labels_train = le.transform(labels_train)
    labels_test = le.transform(labels_test)
    labels_validation = le.transform(labels_validation)

elif args.data == 'tweet50k':
    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()
    le.fit(['abusive', 'normal', 'hateful', 'spam'])

    labels_train = le.transform(labels_train)
    labels_test = le.transform(labels_test)
    labels_validation = le.transform(labels_validation)

else:
    pass



train_labels = torch.tensor(labels_train).to(device)
test_labels = torch.tensor(labels_test).to(device)
validation_labels = torch.tensor(labels_validation).to(device)




if args.BertModel == 'RoBerta':
    from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                             num_labels=NUM_LABELS,
                                                             output_attentions=False,
                                                             output_hidden_states=False)
    print(' ')
    print('using Roberta:')

elif args.BertModel == 'Bert':
    from transformers import BertTokenizer, BertForSequenceClassification
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                          num_labels=NUM_LABELS,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    print(' ')
    print('using Bert:')

elif args.BertModel == 'XLM':
    from transformers import XLMTokenizer, XLMForSequenceClassification, XLMConfig
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-enfr-1024', do_lower_case=True)
    model = XLMForSequenceClassification.from_pretrained('xlm-mlm-enfr-1024',
                                                         num_labels=NUM_LABELS,
                                                         output_attentions=False,
                                                         output_hidden_states=False,
                                                         )
    print(' ')
    print('using XLM:')

elif args.BertModel == 'gpt2':
    from transformers import GPT2Tokenizer, GPT2PreTrainedModel, GPT2DoubleHeadsModel
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
    tokenizer.cls_token = tokenizer.cls_token_id
    tokenizer.pad_token = tokenizer.eos_token
    from gpt2 import GPT2_multiclass_clf

    model = GPT2_multiclass_clf.from_pretrained('gpt2',
                                     num_labels=NUM_LABELS,
                                     output_attentions=False,
                                     output_hidden_states=False,
                                     use_cache=False,
                                     )
    print(' ')
    print('using GPT2:', model_name)
else:
    print('defined multi-class classification but the model fails settingup')


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



def train(model, dataloader):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        b_input_ids = batch[0].long().to(device)
        b_input_mask = batch[1].long().to(device)
        b_labels = batch[2].long().to(device)

        optimizer.zero_grad()

        loss, logit = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            output_attentions = False,
                            #output_hidden_states=False,
                            )

        total_loss += loss.item()

        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    train_loss_this_epoch = total_loss / len(dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(train_loss_this_epoch))
    return train_loss_this_epoch



def validate(model, dataloader):
    print(" === Validate function for multi-class ===")
    model.eval()
    valid_loss, f1_micro_total = 0, 0

    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids = batch[0].long()
        b_input_mask = batch[1].long()
        b_labels = batch[2].long()

        with torch.no_grad():
            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        softmax = torch.nn.functional.softmax(logits, dim=1)
        prediction = softmax.argmax(dim=1).detach().cpu().numpy()

        labels = b_labels.to('cpu').numpy()

        f1_micro = f1_score(labels, prediction, average='micro', zero_division=1)
        f1_micro_total += f1_micro

        valid_loss += loss

    return valid_loss / len(dataloader), f1_micro_total / len(dataloader)
    # Report the final accuracy for this validation run.


def metrics(rounded_preds, label):
    pred_array = rounded_preds.cpu().detach().numpy()
    label_array = label.cpu().detach().numpy()

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


resultname = str(args.BertModel) + '_' + str(args.data)

best_valid_loss = float('inf')
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    t0 = time.time()
    print("")
    print('========== Epoch {:} / {:} =========='.format(epoch_i + 1, epochs))
    train_loss = train(model, train_dataloader)


    print("")
    print("Running Validation...")
    valid_loss, f1_micro = validate(model, validation_dataloader)
    print('validation loss:', valid_loss)
    print('f1_micro:', f1_micro)
    print("  this epoch (training + validation) took: {:}".format(format_time(time.time() - t0)))

#torch.save(model.state_dict(), str(args.resultpath) + resultname + '_model.pt')

print("")
print("Training complete!")

prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, shuffle = False)

model.eval()

predictions = torch.Tensor().to(device)

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        logits = model(b_input_ids.long(), token_type_ids=None, attention_mask=b_input_mask.long())
    softmax = torch.nn.functional.softmax(logits, dim=1)
    prediction = softmax.argmax(dim=1)
    predictions = torch.cat((predictions, prediction.float()))
    # true_labels = torch.cat((true_labels, b_labels.float()))
    print('    DONE.')
predictions_np = predictions.cpu().tolist()
test['prediction'] = predictions_np
test['label_encoded'] = labels_test
f1_micro = f1_score(test['label_encoded'], test['prediction'], average='micro')
f1_macro = f1_score(test['label_encoded'], test['prediction'], average='macro')
print('RESULTS -----------')
print(str(args.data))
print('f1_micro:', f1_micro)
print('f1_macro:', f1_macro)
print(classification_report(test['label_encoded'], test['prediction'], zero_division=1, digits=4))
test.to_csv(str(resultname) + '_result.csv')

