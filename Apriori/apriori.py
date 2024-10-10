# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:34:03 2024

@author: USER
"""

import pandas as pd 
import numpy as np 

df=pd.read_csv("Groceries_dataset.csv")

duplicates = df.duplicated()
print(duplicates.sum())

df.drop_duplicates(inplace=True)


combinanacoes = df.groupby(['Member_number','Date']).agg({
    'itemDescription':list
}).reset_index()

combinacoes_list = combinanacoes['itemDescription'].tolist()
combinacoes_list[0]

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

te=TransactionEncoder()

combool = te.fit_transform(combinacoes_list)
dataset = pd.DataFrame(combool,columns=te.columns_)
print(dataset)


from mlxtend.frequent_patterns import apriori

regras = apriori(dataset,min_support=0.001 ,use_colnames=True)

