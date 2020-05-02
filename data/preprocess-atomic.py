# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
    
import pandas as pd
import json
import pickle

categories = []
categories += ["oEffect"]
categories += ["oReact"]
categories += ["oWant"]
categories += ["xAttr"]
categories += ["xEffect"]
categories += ["xIntent"]
categories += ["xNeed"]
categories += ["xReact"]
categories += ["xWant"]

def zipped_flatten(outer):
    return [(key, fill, el) for key, fill, inner in outer for el in inner]

#Given an event and an inference dimension, aggregate all inferences to compute bleu
for split in ['trn','dev','tst']:
    file_name = "v4_atomic_all_agg.csv"
    df = pd.read_csv("{}/{}".format("atomic", file_name), index_col=0)
    examples=[]
    df=df[df['split']==split]
    df.iloc[:, :9] = df.iloc[:, :9].apply(lambda col: col.apply(json.loads))
    for cat in categories:
        attr = df[cat]
        examples+=zip(attr.index, ["<{}>".format(cat)] * len(attr), attr.values)
    examples=[(x[0],x[1],x[2]) for x in  examples if x[-1]!=[]]
    examples=[(x[0],x[1],[e for e in x[-1] if e!=""]) for x in examples ]
    examples=[x for x in examples if len(x[-1])!=0]  
    pickle.dump(examples,open('atomic/gen-{}.pkl'.format(split),'wb'))

#flatting all examples to train models
for split in ['trn','dev','tst']:
    file_name = "v4_atomic_{}.csv".format(split)
    df = pd.read_csv("{}/{}".format("atomic", file_name), index_col=0)
    examples=[]
    df.iloc[:, :9] = df.iloc[:, :9].apply(lambda col: col.apply(json.loads))
    for cat in categories:
        attr = df[cat]
        examples+=zipped_flatten(zip(attr.index, ["<{}>".format(cat)] * len(attr), attr.values))
    examples=[(e[0],e[1],e[2]) for e in examples]
    examples=[x for x in examples if x[-1]!=""] 
    pickle.dump(examples,open('atomic/ppl-{}.pkl'.format(split),'wb'))
    

