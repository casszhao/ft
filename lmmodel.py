import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
print(device)


''' import trained lm_model'''
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer("./lm_model/vocab.json",
                                  "./lm_model/merges.txt",
                                  )

tokenizer._tokenizer.post_processor = BertProcessing(("</s>", tokenizer.token_to_id("</s>")),
                                                     ("<s>", tokenizer.token_to_id("<s>")),
                                                     )

tokenizer.enable_truncation(max_length=512)


''' star training language model '''
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("./lm_model", max_len=512)

# initialize from a config, not from an existing pretrained model or checkpoint.
from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config)

print(model.num_parameters())
print(model)

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path="./sample_data/sample_data1.txt", block_size=128)
print('created dataset')

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

trainer.train() # what saved are 0. config.json 1. pytorch_model.bin 2. training_args.bin


''' Save final model (+ lm_model + config) to disk '''

trainer.save_model("./lm_model")
print('model saved')

''' check the trained lm'''


