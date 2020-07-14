''' using target datasets for pre-training
including:
         0. preprocessing csv to txt
         1. train LM and save it

'''

import argparse
parser = argparse.ArgumentParser(description='convert json to txt for later training')
# 0
group = parser.add_mutually_exclusive_group()
group.add_argument('--pre_tokenizer', action='store_true', help='need to specify if using pre trained tokenizer from the package')
group.add_argument('--new_tokenizer', action='store_true', help='need to specify if using newly trained tokenizer')
parser.add_argument('--tokenizerfilefolder', type=str, help = 'if using new_tokenizer, need to specify tokenizer file path')

# 1 pre_tokenizer no need
group2 = parser.add_mutually_exclusive_group()
group2.add_argument('--running', action='store_true', help='running using the original big dataset')
group2.add_argument('--testing', action='store_true', help='testing using the small sample.txt dataset')

# 2 text col name
parser.add_argument('--textcolname', type=str)

# 3
parser.add_argument('--txtfile', type=str, help= 'a csv file, to be used for fine-tuning, it should be a concatenated txt file from multiple txt files')
# 4
parser.add_argument('--saved_lm_model', type=str, help= 'where to save the trained language model')
args = parser.parse_args()

import pandas as pd
import regex as re

def basicPreprocess(text):
  try:
    processed_text = text.lower()
    processed_text = re.sub(r'\W +', ' ', processed_text)
  except Exception as e:
    print("Exception:",e,",on text:", text)
    return None
  return processed_text

if args.testing:
    data = pd.read_csv(str(args.txtfile)).sample(10)
elif args.running:
    data = pd.read_csv(str(args.txtfile))

data[str(args.textcolname)] = data[str(args.textcolname)].apply(basicPreprocess).dropna()
data = data[str(args.textcolname)]
data = data.replace('\n', ' ')

with open('listfile.txt', 'w') as filehandle:
    for listitem in data:
        filehandle.write('%s\n' % listitem)

from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM

config = BertConfig(vocab_size=30522,
                    max_position_embeddings=512,
                    num_attention_heads=12,
                    num_hidden_layers=12,
                    type_vocab_size=2,
                    )

if args.new_tokenizer:
    tokenizer = BertTokenizerFast.from_pretrained(str(args.tokenizerfilefolder), max_len=512)
elif args.pre_tokenizer:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)
else:
    print('need to define using new_tokenizer or pre_tokenizer by adding arguments')

model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
print('model parameters:', model.num_parameters())
print(model)


from transformers import LineByLineTextDataset
#paths = [str(x) for x in Path(str(args.txtfolder)).glob("**/*.txt")]
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path= 'listfile.txt', block_size=128)
print('created dataset from folders')

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# initial the trainer

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="lm_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train() # what savedresults are 0. config.json 1. pytorch_model.bin 2. training_args.bin

''' Save final model (+ lm_model + config) to disk '''

trainer.save_model(str(args.saved_lm_model))
#trainer.save_model('./lm_model/')
#reload using from_pretrained()
print('model saved')

''' check the trained lm'''


