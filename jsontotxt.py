import json
import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import regex as re
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='run on multi-label dataset')
parser.add_argument('--jsonfolder', type=str, help='where are those json files')
parser.add_argument('--txtfolder', type=str, help='where to save converted txt files')
args = parser.parse_args()

data = [json.loads(line) for line in open(str(args.jsonfolder), 'r')]
f = open(str(args.txtfolder), 'a+')
for i in data:
     if i['author_flair_text'] != None:
          row = i['author_flair_text']
          print(row)
          f.write("\n")
          f.write(row)
     else:
          pass
f.close()



'''
with open('../sample_data/sample_data.json','r') as json_file:
    for line in json_file.readlines():
         data = json.loads(line)
         print(data['author_flair_text'])

with open('../sample_data/sample_data.json') as json_file:
     for line in json_file.readlines():
          data = json.loads(line)

          file1 = open("MyFile.txt", "a")

          L = data['author_flair_text']
          file1.writelines(L)
          # file1.close()
          print(file1.read())

file = []
for line in open('../sample_data/sample_data.json', 'r'):
     file.append(json.loads(line))


for i in file:
     if i['author_flair_text'] !=None:

          print(i['author_flair_text'])
     else:
          pass


data = [json.loads(line) for line in open('../sample_data/sample_data.json', 'r')]


f = open('testing.txt', 'w')
for i in data:
     if i['author_flair_text'] != None:
          row = i['author_flair_text']
          print(row)
          f.write(row)
     else:
          pass
f.close()
'''
