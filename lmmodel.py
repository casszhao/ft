'''before run lmmodel, using
$ cat /path/to/all/file/dir/* > /path/to/all/file/dir/merges.txt

/path/to/all/file/dir/merges.txt   is the args.txtfile
'''

import torch
from pathlib import Path
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
print(device)


import argparse
parser = argparse.ArgumentParser(description='convert json to txt for later training')
# 0
group = parser.add_mutually_exclusive_group()
group.add_argument('--pre_tokenizer', action='store_true', help='need to specify if using pre trained tokenizer from the package')
group.add_argument('--new_tokenizer', action='store_true', help='need to specify if using newly trained tokenizer')
# 1 pre_tokenizer no need
parser.add_argument('--tokenizerfilefolder', type=str, help = 'if using new_tokenizer, need to specify tokenizer file path')
# 2
parser.add_argument('--txtfile', type=str, help= 'file to be used for fine-tuning, it should be a concatenated txt file from multiple txt files')
# 3
parser.add_argument('--saved_lm_model', type=str, help= 'where to save the trained language model')
args = parser.parse_args()


''' import trained lm_model'''
from tokenizers.implementations import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer
from tokenizers.processors import BertProcessing

# the tokenizer savedresults file, for bert, only one, vocab.txt

''' star training language model '''
from transformers import BertTokenizerFast, BertConfig
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

# initialize from a config, not from an existing pretrained model or checkpoint.
from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
print('model parameters:', model.num_parameters())
print(model)

from transformers import LineByLineTextDataset
#paths = [str(x) for x in Path(str(args.txtfolder)).glob("**/*.txt")]
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path= str(args.txtfile), block_size=128)
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
print('model savedresults, config.json, pytorch_model.bin, training_args.bin, vocab.txt')

''' check the trained lm'''


