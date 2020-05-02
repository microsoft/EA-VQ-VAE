# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import pandas as pd
import json
import pickle

categories = []
categories += ["oReact"]
categories += ["xIntent"]
categories += ["xReact"]

def zipped_flatten(outer):
    return [(key, fill, el) for key, fill, inner in outer for el in inner]

#Given an event and an inference dimension, aggregate all inferences to compute bleu
for split in ['trn.csv','dev.csv','tst.csv']:
    df = pd.read_csv("{}/{}".format("event2mind", split), index_col=1)
    examples=[]
    df.iloc[:, 1:4] = df.iloc[:, 1:4].apply(lambda col: col.apply(json.loads))
    df['xIntent']=df['Xintent']
    df['oReact']=df['Otheremotion']
    df['xReact']=df['Xemotion']
    for cat in categories:
        attr = df[cat]
        examples+=zip(attr.index, ["<{}>".format(cat)] * len(attr), attr.values)
    examples=[(x[0],x[1],x[2]) for x in  examples if x[-1]!=[]]
    dic={}
    for e in examples:
        if (e[0],e[1]) not in dic:
            dic[(e[0],e[1])]=[]
        dic[(e[0],e[1])]+=e[2]  
    examples=[(x[0],x[1],dic[x]) for x in dic]
    examples=[(x[0],x[1],[e for e in x[-1] if e!=""]) for x in examples ]
    examples=[x for x in examples if len(x[-1])!=0]  
    pickle.dump(examples,open('event2mind/gen-{}.pkl'.format(split[:-4]),'wb'))
    
#flatting all examples to train models
for split in ['trn.csv','dev.csv','tst.csv']:
    df = pd.read_csv("{}/{}".format("event2mind", split), index_col=1)
    examples=[]
    df.iloc[:, 1:4] = df.iloc[:, 1:4].apply(lambda col: col.apply(json.loads))
    df['xIntent']=df['Xintent']
    df['oReact']=df['Otheremotion']
    df['xReact']=df['Xemotion']
    for cat in categories:
        attr = df[cat]
        examples+=zipped_flatten(zip(attr.index, ["<{}>".format(cat)] * len(attr), attr.values))
    examples=[(e[0],e[1],e[2]) for e in examples]
    examples=[x for x in examples if x[-1]!=""] 
    pickle.dump(examples,open('event2mind/ppl-{}.pkl'.format(split[:-4]),'wb'))
    

