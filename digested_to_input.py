import pandas as pd
import numpy as np
import pickle
import sys

#input is a digested txt file from protein digestion simulator and output is a pickle file containing filtered input data in format for our prediction model


data = pd.read_csv(sys.argv[1], sep='\t')

df = []
for i in range(len(data)):
    df.append({'Sequence':data.iloc[i]['Sequence'], 'Mass':data.iloc[i]['Monoisotopic_Mass'], 'len':len(data.iloc[i]['Sequence']), 'Protein':data.iloc[i]['Protein_Name']})

print(len(df))
df = pd.DataFrame(df)
df = df[df['Sequence'].str.len() <= 40]
df = df[df['Sequence'].str.len() >= 7]
df = df[df['Sequence'].str.contains('X') == False]
df = df[df['Sequence'].str.contains('U') == False]
df = df[df['Sequence'].str.contains('O') == False]
    
df['Charge'] = 2
df['Modified sequence'] = df['Sequence']
df['Modification'] = ''
print(len(df))

charge3 = df.copy()
charge3['Charge'] = 3    

output = pd.concat([df,charge3])
print(output.head())
print(len(output))

with open(sys.argv[2], 'wb') as f:
    pickle.dump(output,f)
    
