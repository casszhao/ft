'''
https://colab.research.google.com/gist/aditya-malte/2d4f896f471be9c38eb4d723a710768b/smallberta_pretraining.ipynb#scrollTo=Fn68O17MsqYp
'''
import json
from pathlib import Path
import argparse
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
from tokenizers.trainers import BpeTrainer


from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
'''
ByteLevelBPETokenizer just to prevent <unk> tokens entirely. 
Furthermore, the function used to train the tokenizer assumes that each sample is stored in a different text file.
'''
VOCAB_SIZE = 52_000
parser = argparse.ArgumentParser(description='train tokenizer')
parser.add_argument('--txtfolder', type=str, help='the FOLDER where are those txt files')
args = parser.parse_args()

paths = [str(x) for x in Path(str(args.txtfolder)).glob("**/*.txt")]

# Initialize a lm_model
tokenizer = BertWordPieceTokenizer()


#trainer = BpeTrainer(vocab_size= VOCAB_SIZE, show_progress=True, initial_alphabet=ByteLevel.alphabet())
#tokenizer.train(trainer, paths)
# Customize training
'''
tokenizer._tokenizer.post_processor = BertProcessing(("[CLS]", tokenizer.token_to_id("[CLS]")),
                                                     ("[SEP]", tokenizer.token_to_id("[SEP]")),
                                                     )
                                                    '''
tokenizer.train(files=paths, vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["[PAD]",
                                                                                     "[UNK]",
                                                                                     "[CLS]",
                                                                                     "[SEP]",
                                                                                     "[MASK]",
                                                                                     ])
print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
tokenizer.save_model('./lm_model')
print('tokenizer savedresults, they are vocab.json and merges.txt')