#-*- coding:utf-8 -*-
import pandas as pd

import pandas as pd

df = pd.read_csv('data/train.csv')

targets = ['Sex', 'Embarked', 'Survived']

for target in targets:
    df1 = df[target].dropna(how='all').drop_duplicates().reset_index(drop=True)

    csv = '{"":[0'
    for i in range(df1.size):
        csv = csv + ',' + str(0)
    csv = csv + ']'

    for idx, d in enumerate(df1):
        csv = csv + ',\n"' + str(d) + '":[' + str(idx+1)
        for i in range(df1.size):
            if i == idx:
                csv = csv + ',' + str(1)
            else:
                csv = csv + ',' + str(0)
        csv = csv + ']'
    csv = csv + '}'

    with open('dict/'+target+'.json', 'w') as f:
        f.write(csv)






#以下残骸
#print df['Sex'].drop_duplicates()
#df['Sex'].drop_duplicates().reset_index( drop = True ).to_csv('dict/Sex.csv', na_rep='')


#print df['Embarked'].drop_duplicates().reset_index( drop = True )
#df['Embarked'].drop_duplicates().reset_index( drop = True ).to_csv('dict/Embarked.csv', na_rep='')
