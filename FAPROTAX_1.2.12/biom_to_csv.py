import biom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_phylum(otu_id, meta):
    
    tax = meta['taxonomy']
    if len(tax) >= 2 and tax[1] != '':
        return tax[1]
    return 'p__Unknown'

def biom_to_csv_phylum(path):
    
    table = biom.load_table(path)
    
    table_phylum = table.collapse(
    get_phylum, 
    axis='observation',
    norm=False
    )
    
    df_phylum = table_phylum.to_dataframe(dense=True)
    
    X_sb = df_phylum.T
    
    return X_sb
   
def entropy(df):
    
    xi0 = df.div(df.sum(axis=1), axis=0)
    xi1 = df.div(df.sum(axis=0), axis=1)
    
    p0 = xi0[xi0>0]
    p1 = xi1[xi1>0]
    
    Hp = -np.sum(p0 * np.log(p0), axis=0)
    Hs = -np.sum(p1 * np.log(p1), axis=1)
    
    return Hp, Hs
  
