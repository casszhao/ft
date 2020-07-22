''' using target datasets for pre-training
including:
         0. preprocessing csv to txt
         1. train LM and save it
'''

import argparse
parser = argparse.ArgumentParser(description='convert json to txt for later training')
# 0

# 1 pre_tokenizer no need
group = parser.add_mutually_exclusive_group()
group.add_argument('--running', action='store_true', help='running using the original big dataset')
group.add_argument('--testing', action='store_true', help='testing using the small dataset')

# 2 text col name
parser.add_argument('--textcolname', type=str)

parser.add_argument('--num_train_epochs', '-e', type=int)
# 3
parser.add_argument('--csvfile', type=str, help='a csv file, to be used for fine-tuning, it should be a concatenated txt file from multiple txt files')

parser.add_argument('--ftLMpath', type=str, help='a file path which contains pre-trained LM model.')
args = parser.parse_args()

import pandas as pd
import regex as re

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def basicPreprocess(text):
    try:
        processed_text = text.lower()
        processed_text = re.sub(r'\W +', ' ', processed_text)
    except Exception as e:
        print("Exception:", e, ",on text:", text)
        return None
    return processed_text


if args.testing:
    data = pd.read_csv(str(args.csvfile)).sample(1000)
elif args.running:
    data = pd.read_csv(str(args.csvfile))

data[str(args.textcolname)] = data[str(args.textcolname)].apply(basicPreprocess).dropna()
data = data[str(args.textcolname)]
data = data.replace('\n', ' ')

with open(str(args.csvfile) + '.txt', 'w') as filehandle:
    for listitem in data:
        filehandle.write('%s\n' % listitem)

from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM

config = BertConfig(vocab_size=28996,
                    max_position_embeddings=512,
                    num_attention_heads=12,
                    num_hidden_layers=12,
                    type_vocab_size=2,
                    )

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)
model = BertForMaskedLM.from_pretrained(str(args.ftLMpath), config=config)
print('===========================')
print(f'The model (NO frozen paras) has {count_parameters(model):,} trainable parameters')
print('===========================')

params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Transformer 0====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 1====\n')
for p in params[21:37]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 2====\n')
for p in params[37:53]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 3====\n')
for p in params[53:69]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 4====\n')
for p in params[69:85]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 5====\n')
for p in params[85:101]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 6====\n')
for p in params[101:117]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 7====\n')
for p in params[117:133]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 8====\n')
for p in params[133:149]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 9====\n')
for p in params[149:165]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 10====\n')
for p in params[165:181]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer 11====\n')
for p in params[181:197]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Transformer NNNNN====\n')
for p in params[197:-4]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

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


from transformers import LineByLineTextDataset
#paths = [str(x) for x in Path(str(args.txtfolder)).glob("**/*.txt")]
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path= str(args.csvfile) + '.txt', block_size=128)
print('created dataset from folders')

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# initial the trainer

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    do_train=True,
    do_predict=True,
    output_dir=str(args.ftLMpath) + '_' + str(args.csvfile)+'_LMmodel',
    overwrite_output_dir=True,
    num_train_epochs= args.num_train_epochs,
    per_device_train_batch_size=32,
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

trainer.train()
''' Save final model (+ lm_model + config) to disk '''

trainer.save_model(str(args.ftLMpath) + '_' + str(args.csvfile)+'_LMmodel')
print('the language model saved as ', str(args.csvfile)+'_LMmodel')

''' check the trained lm'''

