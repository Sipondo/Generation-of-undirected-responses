################
#Translates raw reddit comment data files (60-80 gigabytes per) into .csv
#files containing the comments from a specific subreddit
#Written in python 2 i.o. python 3.
#
#All code is strictly our work; Bauke Brenninkmeijer and Ties Robroek.


import json
from tqdm import tqdm
import mmap
import unicodecsv as csv

file_path = "2018_02_raw/RC_2018-02"

def get_num_lines(file_path):
    return sum(1 for line in open(file_path))

file = open(file_path)
line = file.readline()
jayson = json.loads(line)
keylist = jayson.keys()
file.close()


print_path = "export.csv"
with open(file_path) as file:
    with open(print_path, 'w') as export_file:
        writer =  csv.DictWriter(export_file, fieldnames = keylist)
        writer.writeheader()
        for line in tqdm(file, total=get_num_lines(file_path)):
            content = json.loads(line)
            if u"subreddit" in content:
                if content[u"subreddit"] == "AskReddit":#"explainlikeimfive":
                    writing_content = {value: content[value] for value in content if value in keylist} #for key in keylist
                    writer.writerow(writing_content)
