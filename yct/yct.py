import requests
import json
import csv

f_write = open('train_data3.tsv', 'wt')
writer = csv.writer(f_write, delimiter='\t')

url = 'http://shortcuts.yahooapis.com:4080/annotate'
with open('clean_all2.txt', 'r') as f:
    for line in f:
        try:
            keyword__my_title = line.strip('\n')
            my_title = keyword__my_title.split('\t')[1]
            myobj = {'text': my_title}

            response = json.loads(requests.post(url, data = myobj).text)
            cate = response['document']['categories']['yct']['categories'][0]['category_name']
            writer.writerow([my_title, cate])
        except:
            continue
