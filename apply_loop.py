'''apply fine-tuned bert based modle on four datasets'''

from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

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
parser.add_argument('--FTModel', type=str, help= 'where is the saved trained language model, including path and name')
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

if args.data == 'multi-label':
    NUM_LABELS = 6
elif args.data == 'AG10K':
    NUM_LABELS = 3
else:
    NUM_LABELS = 4

batch_size = 16
epochs = args.epochs




def train_multiclass(model, dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):

        if step % 2000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

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

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    train_loss_this_epoch = total_loss / len(dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(train_loss_this_epoch))
    return train_loss_this_epoch



def validate_multiclass(model, dataloader):
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

print(f'The model (NO frozen paras) has {count_parameters(model):,} trainable parameters')

print('===========================')
print(f'The model has {count_parameters(model):,} trainable parameters')
print('===========================')

model.to(device)










# except AG10K having a validation set, other datasets need to be divided
train_path = './data/' + str(args.data) + '.csv'
print('train path:', train_path)


if args.data == 'AG10K':
    if args.testing:
        data = pd.read_csv(train_path, names = ['id', 'comment', 'label']).sample(5000)
    elif args.running:
        data = pd.read_csv(train_path, names = ['id', 'comment', 'label'])
    else:
        print('need to define parameter, it is "--running" or "--testing"')
else:
    if args.testing:
        data = pd.read_csv(train_path).sample(5000)
    elif args.running:
        data = pd.read_csv(train_path)
    else:
        print('need to define parameter, it is "--running" or "--testing"')



def one_loop(percent):
    if args.data == 'AG10K':
        train, test = train_test_split(data, test_size=0.2, stratify=data['label'])
        validation = pd.read_csv('./data/' + str(args.data) + '_dev.csv', names=['id', 'comment', 'label']).dropna()
    elif args.data == 'multi-label':
        train, test = train_test_split(data, test_size=0.2, stratify=data['severe_toxic'])
        test, validation = train_test_split(data, test_size=0.5, stratify=data['severe_toxic'])
    else:
        train, test = train_test_split(data, test_size=0.2, stratify=data['label'])
        test, validation = train_test_split(data, test_size=0.5, stratify=data['label'])

    dispose_p = (100 - percent)/100
    print('--------------training dataset size: ', 1 - dispose_p)
    if args.data == 'multi-label':
        train, dispose = train_test_split(data, test_size=dispose_p, stratify=data['severe_toxic'])
    else:
        train, dispose = train_test_split(data, test_size= dispose_p, stratify=data['label'])

    if args.data == 'multi-label':
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
    elif args.data == 'wassem' or 'AG10K' or 'tweet50k':
        sentences_train = train.comment.values
        labels_train = train.label.values

        sentences_test = test.comment.values
        labels_test = test.label.values

        sentences_validation = validation.comment.values
        labels_validation = validation.label.values
    else:
        print('the definition of args.data is invalid')


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



    # put train_labels to tensor and device
    if args.data == 'multi-label':
        pass
    else:
        train_labels = torch.tensor(labels_train).to(device)
        test_labels = torch.tensor(labels_test).to(device)
        validation_labels = torch.tensor(labels_validation).to(device)


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


    '''
    ================== Training Loop =======================
    '''
    optimizer = AdamW(model.parameters(), lr=0.0005, weight_decay = 0.01, eps = 1e-6)
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps= int(total_steps*0.06),  # Default value in run_glue.py
                                                num_training_steps=total_steps)


    if args.FTModel != None:
        resultname = str(args.FTModel)
    else:
        resultname = str(args.BertModel) + '_' + str(args.data)

    best_valid_loss = float('inf')
    loss_values = []

    # For each epoch...
    for epoch_i in range(0, epochs):
        print("")
        print('========== Epoch {:} / {:} =========='.format(epoch_i + 1, epochs))
        t0 = time.time()
        if args.data == 'multi-label':
            from multi_label_fns import train_multilabel, validate_multilable
            train_loss = train_multilabel(model, train_dataloader)
        else:
            train_loss = train_multiclass(model, train_dataloader, optimizer, scheduler)
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        print("")
        print("Running Validation...")

        t0 = time.time()
        if args.data == 'multi-label':
            valid_loss, micro_f1 = validate_multilable(model, validation_dataloader)
        else:
            valid_loss, micro_f1 = validate_multiclass(model, validation_dataloader)
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        print('validatin loss:  ', valid_loss)
        print('micro F1: ', micro_f1)

    #torch.save(model.state_dict(), str(args.resultpath) + resultname + '_model.pt')

    print("")
    print("Training complete!")

    prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, shuffle = False)

    model.eval()

    predictions = torch.Tensor().to(device)


    if args.data == 'multi-label':
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
        result.to_csv(str(args.resultpath) + model_name +str(args.freeze) + '_result.csv', sep='\t')

    else:
        for batch in prediction_dataloader:
            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                # Forward pass, calculate logit predictions, 没有给label, 所以不outputloss
                outputs = model(b_input_ids.long(), token_type_ids=None,
                                attention_mask=b_input_mask)  # return: loss(only if label is given), logit
            logits = outputs[0]
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
        print('f1_micro:', f1_micro, 'f1_macro:', f1_macro)
        print(classification_report(test['label_encoded'], test['prediction'], zero_division=1, digits=4))

# 5, 10, 15
for i in range(1, 20):
    percent = i*5
    print('---------------------------  percent:  ', percent)
    one_loop(percent)


