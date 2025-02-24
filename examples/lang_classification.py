# This file is temporarily bad
# It will be updated to be more readable soon

import hypervectors as hv

import pandas as pd
import numpy as np
import string
import unicodedata
import re

import tensorflow as tf

train_no = 256   # TODO: must be > 3

def clean_string(input_string):
    # convert to lowercase
    input_string = input_string.lower()
    
    # remove diacritics (accented characters)
    input_string = ''.join(
        c for c in unicodedata.normalize('NFD', input_string) if unicodedata.category(c) != 'Mn'
    )
    
    # replace spaces with the '#' character
    input_string = input_string.replace(' ', '#')

    input_string = re.sub(r'[^a-z#]', '', input_string)
    
    return input_string

def to_ngrams(in_str, ngram_len):
    ret = []

    for i in range(len(in_str) - ngram_len + 1):
        ret.append(in_str[i:i + ngram_len])

    return ret

splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["train"])

df_filtered = df[df['labels'].isin(['en', 'fr', 'de'])]
print(df_filtered)

df_filtered_shuffled = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
df_filtered_shuffled = df_filtered_shuffled.head(train_no)

### this is stupid right now...
df_filtered_en = df_filtered[df_filtered['labels'] == 'en']
df_filtered_shuffled_en = df_filtered_en.sample(frac=1, random_state=42).reset_index(drop=True)

df_filtered_fr = df_filtered[df_filtered['labels'] == 'fr']
df_filtered_shuffled_fr = df_filtered_fr.sample(frac=1, random_state=42).reset_index(drop=True)

df_filtered_de = df_filtered[df_filtered['labels'] == 'de']
df_filtered_shuffled_de = df_filtered_de.sample(frac=1, random_state=42).reset_index(drop=True)
### ...

alphabet_en = list(string.ascii_lowercase)

syms = alphabet_en
syms.append("#")

hv_size = 256

sym_HVs = hv.gen_P_HVs(syms, random_method="tf_random", hv_size=hv_size)

# 3500/500 train/test per language
 
ngram_len = 3

for row in df_filtered_shuffled_en.itertuples():
    str = clean_string(row.text)
    n_grams = (to_ngrams(str, ngram_len=3))

    print(n_grams)

for row in df_filtered_shuffled_fr.itertuples():
    str = clean_string(row.text)
    n_grams = (to_ngrams(str, ngram_len=3))

    print(n_grams)

for row in df_filtered_shuffled_de.itertuples():
    str = clean_string(row.text)
    n_grams = (to_ngrams(str, ngram_len=3))

    print(n_grams)

