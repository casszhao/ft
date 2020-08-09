''' using target datasets for pre-training
including:
         0. preprocessing csv to txt
         1. train LM and save it
'''

import torch

import argparse
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


parser = argparse.ArgumentParser(description='convert json to txt for later training')
parser.add_argument('--num_train_epochs', '-e', type=int)
parser.add_argument('--LM', type=str, action='store', choices = ['Bert','RoBerta','XLM'])
args = parser.parse_args()

import pandas as pd
import regex as re

MAX_LEN = 100
batch_size = 16
epochs = 1


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


train = pd.read_csv('multi-label_testing_train.csv')
test = pd.read_csv('multi-label_testing_test.csv')
validation = pd.read_csv('multi-label_testing_test.csv')

sentences_train = train.comment_text.values
sentences_test = test.comment_text.values
sentences_validation = validation.comment_text.values

if args.LM == 'Bert':
    from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM

    config = BertConfig(vocab_size=28996,
                        max_position_embeddings=512,
                        num_attention_heads=12,
                        num_hidden_layers=12,
                        #type_vocab_size=2, default is 2
                        )
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)
    model = BertForMaskedLM.from_pretrained('bert-base-cased', config=config)
    # 12-layer, 768-hidden, 12-heads, 110M parameters.

elif args.LM == 'RoBerta':
    from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM

    config = RobertaConfig(vocab_size=50265,
                           max_position_embeddings=514,
                           num_attention_heads=12,
                           num_hidden_layers=12,
                           type_vocab_size=1,
                           )
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=False)
    model = RobertaForMaskedLM.from_pretrained('roberta-base', config=config)
    # 12-layer, 768-hidden, 12-heads, 125M parameters, roberta-base using the bert-base architecture

elif args.LM == 'XLM':
    from transformers import XLMConfig, XLMTokenizer, XLMWithLMHeadModel
    '''
    config = XLMConfig(vocab_size=95000,
                       emb_dim=1024,
                       max_position_embeddings=512,
                       n_heads=16,
                       n_layers=12,
                       )
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024', do_lower_case=False)
    model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-xnli15-1024', config=config)
    # 12-layer, 1024-hidden, 8-heads
    # XLM Model pre-trained with MLM on the 15 XNLI languages.
    '''
    config = XLMConfig(vocab_size=64139,
                       emb_dim=1024,
                       max_position_embeddings=512,
                       n_heads=8,
                       n_layers=6,
                       )

    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-enfr-1024', do_lower_case=False)
    model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-enfr-1024', config=config)
    #6-layer, 1024-hidden, 8-heads
    # XLM English-French model trained on the concatenation of English and French wikipedia

else:
    print('need to define LM from Bert,RoBerta,XLM')


'''
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, AdamW

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)
'''

train_inputs = torch.Tensor()
train_masks = torch.Tensor()
for sent in sentences_train:
    encoded_sent = tokenizer.encode_plus(sent,  # Sentence to encode.
                                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                         max_length=MAX_LEN,  # Truncate all sentences.
                                         truncation=True,
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         return_token_type_ids=False,
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
                                         truncation=True,
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         return_token_type_ids=False,
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
                                         truncation=True,
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         return_token_type_ids=False,
                                         return_tensors='pt')  # return pytorch not tensorflow tensor
    test_inputs = torch.cat((test_inputs, encoded_sent['input_ids'].float()), dim=0)
    test_masks = torch.cat((test_masks, encoded_sent['attention_mask'].float()), dim=0)

test_inputs.to(device)
test_masks.to(device)

# for training data
train_data = TensorDataset(train_inputs, train_masks)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# for validation set.
validation_data = TensorDataset(validation_inputs, validation_masks)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


'''
config = BertConfig(vocab_size=28996,
                    max_position_embeddings=512,
                    num_attention_heads=12,
                    num_hidden_layers=12,
                    type_vocab_size=2,
                    )
model = BertForMaskedLM.from_pretrained('bert-base-cased', config=config)
'''

optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

for epoch_i in range(0, epochs):
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].long().to(device)
        b_input_mask = batch[1].long().to(device)

        optimizer.zero_grad()
        loss = model(b_input_ids,
                     token_type_ids=None,
                     attention_mask=b_input_mask,
                     labels=b_input_ids,
                     )[0]

        loss.backward()
        optimizer.step()


torch.save(model.state_dict(), str(args.resultpath) + str(args.BertModel) + '_multi-label.pt')

print("")
print("Training complete!")



