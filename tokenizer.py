'''
https://colab.research.google.com/gist/aditya-malte/2d4f896f471be9c38eb4d723a710768b/smallberta_pretraining.ipynb#scrollTo=Fn68O17MsqYp
'''
import json
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer


'''
# for processing the list derived from a dataframe
def basicPreprocess(text):
  try:
    processed_text = text.lower()
    processed_text = re.sub(r'\W +', ' ', processed_text)
  except Exception as e:
    print("Exception:",e,",on text:", text)
    return None
  return processed_text
'''

paths = [str(x) for x in Path("sample_data/").glob("**/*.txt")]

# Initialize a lm_model
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>",
                                                                                 "<mask>"])
tokenizer.save_model('lm_model')