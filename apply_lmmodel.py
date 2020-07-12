from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
import torch
import torch.nn as nn
import time
from transformers import AdamW
import pandas as pd
from sklearn.metrics import f1_score
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


parser = argparse.ArgumentParser(description='run fine-tuned model on multi-label dataset')
# parser.add_argument('trainable', type=str, action='store', choices = ['fix','nofix'])
# 1
parser.add_argument('saved_lm_model', type=str, help= 'where to save the trained language model')
# 2
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='', help='how many epochs')
# 3
group = parser.add_mutually_exclusive_group()
group.add_argument('--running', action='store_true', help='running using the original big dataset')
group.add_argument('--testing', action='store_true', help='testing using the small sample.txt dataset')
# 4
parser.add_argument('--resultpath', type=str, help='where to save the result csv')
args = parser.parse_args()


MAX_LEN = 100
NUM_LABELS = 6

batch_size = 16
epochs = args.epochs



def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


if args.testing:
    train_path = 'multi-label_testing_train.csv'
    test_path = 'multi-label_testing_test.csv'
    validation_path = 'multi-label_testing_valid.csv'
elif args.running:
    train_path = 'multi-label_train.csv'
    test_path = 'multi-label_test.csv'
    validation_path = 'multi-label_validation.csv'
else:
    print('need to define parameter, it is "--running" or "--testing"')


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
validation = pd.read_csv(validation_path)

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


#config = RobertaConfig.from_json_file('./ft/lm_model/config.json')
#print('loaded config')
# model = RobertaConfig.from_pretrained(pretrained_model_name_or_path = './ft/lm_model', from_tf=False, config=config)
tokenizer = BertTokenizerFast.from_pretrained(str(args.saved_lm_model), max_len=512)


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
'''
from transformers import RobertaTokenizer
lm_model = RobertaTokenizer.from_pretrained('distilroberta-base', do_lower_case=False)
'''

model = Bert_clf.from_pretrained(str(args.saved_lm_model),
                                 num_labels=NUM_LABELS,
                                 output_attentions=False,
                                 output_hidden_states=True)

print(model)

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

        if step % 2000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

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
    loss_values.append(loss)

    print("")
    print("  Average training loss: {0:.2f}".format(train_loss_this_epoch))
    return train_loss_this_epoch


def validate(model, dataloader):
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

    return valid_loss / len(dataloader)
    # Report the final accuracy for this validation run.
    print("  F1 Micro: {0:.2f}".format(f1_micro_total / len(dataloader)))


def metrics(preds, label):
    """
    preds (batch size, 6) before sigmoid
    label (batch size, 6)
    """
    rounded_preds = torch.round(torch.sigmoid(preds))  # (batch size, 6)
    pred_array = rounded_preds.cpu().detach().numpy()
    label_array = label.cpu().detach().numpy()

    correct = (rounded_preds == label).float()  # convert into float for division
    acc = correct.sum() / len(correct)

    micro_f1 = f1_score(label_array, pred_array, average='micro', zero_division=1)
    macro_f1 = f1_score(label_array, pred_array, average='macro', zero_division=1)
    return acc, micro_f1, macro_f1

'''
================== Training Loop =======================
'''
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

best_valid_loss = float('inf')
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    print("")
    print('========== Epoch {:} / {:} =========='.format(epoch_i + 1, epochs))
    t0 = time.time()
    train_loss = train(model, train_dataloader)
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Running Validation...")

    t0 = time.time()
    valid_loss = validate(model, validation_dataloader)
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    '''
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_los
        '''
torch.save(model.state_dict(), str(args.resultpath) + 'Bert_ft_multi-label_model.pt')

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
    logits = outputs
    rounded_preds = torch.round(torch.sigmoid(logits))
    predictions = torch.cat((predictions, rounded_preds.float()))
    labels = torch.cat((labels, b_labels.float()))
print('    DONE.')

acc, f1_micro, f1_macro = metrics(predictions, labels)
print("acc is {}, micro is {}, macro is {}".format(acc, f1_micro, f1_macro))
predictions_np = predictions.cpu().numpy()
predictions_df = pd.DataFrame(predictions_np,
                              columns = ['pred_toxic', 'pred_severe_toxic', 'pred_obscene', 'pred_threat', 'pred_insult', 'pred_identity_hate'])

result = pd.concat([test, predictions_df], axis=1)

f1_toxic = f1_score(result['toxic'], result['pred_toxic'])
f1_severe_toxic = f1_score(result['severe_toxic'], result['pred_severe_toxic'])
f1_obscene = f1_score(result['obscene'], result['pred_obscene'])
f1_threat = f1_score(result['threat'], result['pred_threat'])
f1_insult = f1_score(result['insult'], result['pred_insult'])
f1_identity_hate = f1_score(result['identity_hate'], result['pred_identity_hate'])
print("f1_toxic:", f1_toxic)
print("f1_severe_toxic:", f1_severe_toxic)
print("f1_threat:", f1_threat)
print("f1_obscene:", f1_obscene)
print("f1_insult:", f1_insult)
print("f1_identity_hate:", f1_identity_hate)
print("macro F1:", (f1_toxic + f1_severe_toxic + f1_obscene + f1_threat + f1_insult + f1_identity_hate)/6)

result.to_csv(str(args.resultpath) + 'ft_result.csv', sep='\t')

