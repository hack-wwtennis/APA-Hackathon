import os
import pandas as pd
import numpy as np
import pickle as pkl

#data_directory = os.path.join('.')
#data_filename = r'Parvo Data.xlsx'

#data = pd.read_excel(os.path.join(data_directory, data_filename), sheet_name='Treatment Data')
#outcomes_data = pd.read_excel(os.path.join(data_directory, data_filename), sheet_name='Parvo Admission Data')

with open('data.pkl', 'rb') as f:
    df = pkl.load(f)

    rows, cols = df.shape
    
    curA = ''
    df2 = pd.DataFrame(columns=df.columns)
    for r in range(rows):
        a = df.iloc[r]['Dog A#']
        if a == curA:
            df2.loc[len(df2)-1]['outcome'] = df.iloc[r]['outcome']
        else:
            df2.loc[len(df2)] = df.iloc[r]
            curA = a
    
    # print(df2)
    
    writer = pd.ExcelWriter('output2.xlsx')
    df2.to_excel(writer,'Sheet1')
    writer.save()
    #for c in range(cols):
    #    d = {}
    #    for r in range(rows):
    #        data = str(df.iloc[r][c])
    #        if data in d:
    #            d[data] += 1
    #        else:
    #            d[data] = 1
    #    print(df.columns[c], end=' ')
    #    if len(d) < 50:
    #        print(d)
    #    else:
    #        print(len(d), ' elements')



    #print(data2)
    #print('abc',type(data2))

