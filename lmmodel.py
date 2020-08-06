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

data = pd.read_csv('multi-label_train.csv')

data['comment_text'] = data['comment_text'].apply(basicPreprocess).dropna()
data = data['comment_text']
data = data.replace('\n', ' ')

with open('multi-label_train.csv.txt', 'w') as filehandle:
    for listitem in data:
        filehandle.write('%s\n' % listitem)


