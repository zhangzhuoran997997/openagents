import json
from tqdm import tqdm
from collections import defaultdict, Counter

def write_data(path, data, mode='json'):
    with open(path, 'w') as fout:
        for line in data:
            if mode == 'json':
                fout.write("%s\n" % json.dumps(line, ensure_ascii=False))
            else:
                fout.write("%s\n" % line)

def get_data(path, mode='json'):
    data = []
    with open(path, 'r') as src:
        for line in tqdm(src):
            if mode == 'json':
                line = json.loads(line)
            else:
                line = line.split("\n")[0]
            data.append(line)
    return data

def dump_json(path, data): 
    with open (path,'w') as f:
        json.dump(data,f)
        
def get_json(path): 
    with open (path,'r') as src:
        return json.load(src)