import numpy as np
import pandas as pd


with open('bad_files.txt','rb') as myFile:
    a = myFile.readlines()

b = pd.read_csv('../../../all_extracted/train_shorter.csv')

print(b)
print(len(b)) 

for j in a:
    #print(j.decode("utf-8").strip()) 
    ind = b[b.file_path==j.decode("utf-8").strip()].index
    print(ind)
    b=b.drop(ind)

for j in a:
    ind = b[b.file_path==j.decode("utf-8").strip()].index
    print(ind)
    b.drop(ind)

print(len(b))
data = {'ind':[str(x) for x in range(len(b.file_path))], 'file_path': b.file_path, 'length': b.length}
df = pd.DataFrame(data, columns = ['ind', 'file_path', 'length'])
df.to_csv('new_train_shorter.csv', index=False, header=True)
    
