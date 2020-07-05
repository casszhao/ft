import json
import argparse

parser = argparse.ArgumentParser(description='convert json to txt for later training')
parser.add_argument('--txtfolder', type=str, help='the FOLDER where are those txt and json files')
parser.add_argument('--jsonfilename', type=str, help='json file name within the txt folder')
parser.add_argument('--txtfilename', type=str, help='file name for the txt file to be created within the txt folder')
args = parser.parse_args()

data = [json.loads(line) for line in open(str(args.txtfolder)+str(args.jsonfilename), 'r')]
f = open(str(args.txtfolder)+str(args.txtfilename), 'a+')
for i in data:
     if i['author_flair_text'] != None:
          row = i['author_flair_text']
          print(row)
          #f.write("\n")
          f.write(row)
     else:
          pass
f.close()
