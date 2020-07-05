'''
https://colab.research.google.com/gist/aditya-malte/2d4f896f471be9c38eb4d723a710768b/smallberta_pretraining.ipynb#scrollTo=Fn68O17MsqYp
'''
import json
from pathlib import Path
import argparse
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer


parser = argparse.ArgumentParser(description='train tokenizer')
parser.add_argument('--txtfolder', type=str, help='the FOLDER where are those txt files')
args = parser.parse_args()

paths = [str(x) for x in Path(str(args.txtfolder)).glob("**/*.txt")]

# Initialize a lm_model
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>",
                                                                                 "<mask>"])
tokenizer.save_model('lm_model')
print('tokenizer saved, they are merges.txt and vocab.json')