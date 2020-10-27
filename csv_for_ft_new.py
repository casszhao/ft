import torch
import argparse
import time
import datetime
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


parser = argparse.ArgumentParser(description='this version use already made txt file')
# 0
parser.add_argument('--LM', type=str, action='store', choices = ['Bert','RoBerta','XLM'])

# 1
group = parser.add_mutually_exclusive_group()
group.add_argument('--running', action='store_true', help='running using the original big dataset')
group.add_argument('--testing', action='store_true', help='testing using the small dataset')

# 2 epoch and batch size
parser.add_argument('--num_train_epochs', '-e', type=int)
parser.add_argument('--batch_size', '-b', type=int)

# 3
parser.add_argument('--data', type=str, action='store')

# 4
parser.add_argument('--resultpath', type=str, help='where to save the LM model')
args = parser.parse_args()

import pandas as pd
import regex as re

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    config = XLMConfig(vocab_size=64139,
                       emb_dim=1024,
                       max_position_embeddings=512,
                       n_heads=8,
                       n_layers=6,
                       )

    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-enfr-1024', do_lower_case=False)
    model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-enfr-1024', config=config)
    # 6-layer, 1024-hidden, 8-heads
    # XLM English-French model trained on the concatenation of English and French wikipedia

else:
    print('need to define LM from Bert,RoBerta,XLM')

print(model)

print('===========================')
print('The model has: ', count_parameters(model))
print('===========================')

if args.testing:
    file_path = 'xaa.txt'
else:
    file_path = str(args.data) + '_train.csv.txt'
print('file_path: ', file_path)

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=128)
#dataset = load_dataset("./csv_for_ft_new.py", data_files=file_path)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)


dir = str(args.resultpath) + str(args.data) + '_' + str(args.LM) + '_e' + str(args.num_train_epochs) + '_b' + str(args.batch_size)

training_args = TrainingArguments(
    do_train=True,
    do_predict=True,
    output_dir=dir,
    overwrite_output_dir=True,
    num_train_epochs= args.num_train_epochs,
    per_device_train_batch_size=args.batch_size,
    save_steps=1000,
    save_total_limit=2,
)

# default learning rate(5e-5)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


t0 = time.time()
trainer.train()
elapsed = format_time(time.time() - t0)
print('============= training LM time: ', elapsed)


''' Save final model (+ lm_model + config) to disk '''

trainer.save_model(dir)
print('the language model saved as ', dir)

''' check the trained lm'''


