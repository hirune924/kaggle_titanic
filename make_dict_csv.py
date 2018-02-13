#-*- coding:utf-8 -*-
import pandas as pd

import pandas as pd

df = pd.read_csv('data/train.csv')

targets = ['Sex', 'Embarked']

for target in targets:
    df1 = df[target].drop_duplicates().reset_index(drop=True)

    csv = ''
    for idx, d in enumerate(df1):
        csv = csv + str(d) + ',' + str(idx) + ','
        for i in range(df1.size):
            if i == idx:
                csv = csv + str(1) + ','
            else:
                csv = csv + str(0) + ','
        csv = csv + '\n'

    with open('dict/'+target+'.csv', 'w') as f:
        f.write(csv)






#以下残骸
#print df['Sex'].drop_duplicates()
#df['Sex'].drop_duplicates().reset_index( drop = True ).to_csv('dict/Sex.csv', na_rep='')


#print df['Embarked'].drop_duplicates().reset_index( drop = True )
#df['Embarked'].drop_duplicates().reset_index( drop = True ).to_csv('dict/Embarked.csv', na_rep='')