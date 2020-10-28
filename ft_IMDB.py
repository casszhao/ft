import torch
import argparse
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os

parser = argparse.ArgumentParser(description='fine-tune roberta using IMDB')

parser.add_argument('--epochs', '-e', type=int)
parser.add_argument('--savedname', type=str, help='the file to save the model')

args = parser.parse_args()

savedname = args.savedname
os.makedirs('./' + savedname)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

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

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='IMDB_train.csv.txt', block_size=128)
#dataset = load_dataset("./csv_for_ft_new.py", data_files=file_path)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    do_train=True,
    do_predict=True,
    output_dir=savedname,
    overwrite_output_dir=True,
    num_train_epochs= args.epochs,
    per_device_train_batch_size=16,
    save_steps=1000,
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
trainer.save_model(savedname)
print('the language model saved as ', savedname)