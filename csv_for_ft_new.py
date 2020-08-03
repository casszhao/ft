''' using target datasets for pre-training
including:
         0. preprocessing csv to txt
         1. train LM and save it
'''

import argparse
import time
import datetime

parser = argparse.ArgumentParser(description='this version use already made txt file')
# 0
parser.add_argument('--LM', type=str, action='store', choices = ['Bert','RoBerta','XLM'])

# 1 pre_tokenizer no need
group = parser.add_mutually_exclusive_group()
group.add_argument('--running', action='store_true', help='running using the original big dataset')
group.add_argument('--testing', action='store_true', help='testing using the small dataset')

# 2 text col name
#parser.add_argument('--textcolname', type=str)

parser.add_argument('--num_train_epochs', '-e', type=int)
# 3
parser.add_argument('--data', type=str, action='store', choices = ['AG10K', 'wassem', 'tweet50k', 'multi-label'])

# 4
parser.add_argument('--resultpath', type=str, help='where to save the LM model')
args = parser.parse_args()

import pandas as pd
import regex as re

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


'''  processing csv to txt, now already had the txt file, no need to process it every time
if args.testing:
    data = pd.read_csv(str(args.csvfile)).sample(10)
elif args.running:
    data = pd.read_csv(str(args.csvfile))

def basicPreprocess(text):
    try:
        processed_text = text.lower()
        processed_text = re.sub(r'\W +', ' ', processed_text)
    except Exception as e:
        print("Exception:", e, ",on text:", text)
        return None
    return processed_text
    
data[str(args.textcolname)] = data[str(args.textcolname)].apply(basicPreprocess).dropna()
data = data[str(args.textcolname)]
data = data.replace('\n', ' ')


with open(str(args.csvfile) + '.txt', 'w') as filehandle:
    for listitem in data:
        filehandle.write('%s\n' % listitem)
        '''

if args.LM == 'Bert':
    from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM
    config = BertConfig(vocab_size=28996,
                        max_position_embeddings=512,
                        num_attention_heads=12,
                        num_hidden_layers=12,
                        type_vocab_size=2,
                        )
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)
    model = BertForMaskedLM.from_pretrained('bert-base-cased', config=config)

elif args.LM == 'RoBerta':
    from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM
    config = RobertaConfig(
        vocab_size=50265,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=False)
    model = RobertaForMaskedLM.from_pretrained('roberta-base' ,config=config)

elif args.LM == 'XLM':
    from transformers import XLMConfig, XLMTokenizer, XLMWithLMHeadModel
    config = XLMConfig(
        vocab_size=64139,
        emb_dim = 1024,
        max_position_embeddings=512,
        n_heads = 16,
        n_layers = 12,
    )

    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-enfr-1024', do_lower_case=False)
    model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-enfr-1024', config=config)

else:
    print('need to define LM from Bert,RoBerta,XLM')


print('===========================')
print(f'The model (NO frozen paras) has {count_parameters(model):,} trainable parameters')
print('===========================')

def freeze_layer_fun(freeze_layer):
    for name, param in model.named_parameters():
        if freeze_layer in name:
            print(name)
            param.requires_grad = False
        else:
            pass

print('++++++++++++++++++++++++++++++++++++ freeze layers +++++++++++++++++++++++++++++')
'''
freeze_layer_1 = '.1.'
freeze_layer_fun(freeze_layer_1)

freeze_layer_2 = '.2.'
print(variable.grad)

print(variable.data)
print(variable.data.numpy())
freeze_layer_fun(freeze_layer_2)

freeze_layer_3 = '.3.'
freeze_layer_fun(freeze_layer_3)

freeze_layer_4 = '.4.'
freeze_layer_fun(freeze_layer_4)

freeze_layer_5 = '.5.'
freeze_layer_fun(freeze_layer_5)

freeze_layer_6 = '.6.'
freeze_layer_fun(freeze_layer_6)

freeze_layer_7 = '.7.'
freeze_layer_fun(freeze_layer_7)

freeze_layer_8 = '.8.'
freeze_layer_fun(freeze_layer_8)

freeze_layer_9 = '.9.'
freeze_layer_fun(freeze_layer_9)

freeze_layer_10 = '.10.'
freeze_layer_fun(freeze_layer_10)
'''
print('===========================')
print(f'The model now has {count_parameters(model):,} trainable parameters')
print('===========================')


from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
#paths = [str(x) for x in Path(str(args.txtfolder)).glob("**/*.txt")]
file_path = str(args.data) + '_train.csv.txt'
print('file_path: ', file_path)
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path= file_path, block_size=128)
print('created dataset for LineByLineTextDataset() from folders')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# initial the trainer

from transformers import Trainer, TrainingArguments

dir = str(args.resultpath) + str(args.data) + '_' + str(args.LM) + '_e' + str(args.num_train_epochs)

training_args = TrainingArguments(
    do_train=True,
    do_predict=True,
    output_dir=dir,
    overwrite_output_dir=True,
    num_train_epochs= args.num_train_epochs,
    per_device_train_batch_size=16,
    save_steps=1_000,
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

'''
training_args = TrainingArguments(
    do_train=True,
    do_predict=True,
    output_dir=dir,
    overwrite_output_dir=True,
    num_train_epochs= args.num_train_epochs,
    per_device_train_batch_size=16,
    save_steps=1_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)
'''

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


