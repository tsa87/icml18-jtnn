import pandas as pd
  
# Then loading csv file
df = pd.read_csv('qm9.csv') 
# converting ;FRUIT_NAME' column into list
a = list(df['smiles'])
# another way for joining used
# printing result
with open('vocab.txt', 'w') as f:
    for item in a:
        f.write("%s\n" % item)
