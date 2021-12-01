#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Trent Hannack
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import os
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
import streamlit as st

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

st.title("MABA 6490 Assignment 4")

stopwords=list(STOP_WORDS)
from string import punctuation
punctuation=punctuation+ '\n'

import pandas as pd
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle as pkl
#import os

embedder = SentenceTransformer('all-MiniLM-L6-v2')
  
df = pd.read_csv('Hotel New York Combined.csv')

df['hotel_name'].drop_duplicates()

df_combined = df.sort_values(['hotel_name']).groupby('hotel_name', sort=False).review_body.apply(''.join).reset_index(name='all_review')

import re

df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))

df = df_combined

df_sentences = df_combined.set_index("all_review")

df_sentences = df_sentences["hotel_name"].to_dict()
df_sentences_list = list(df_sentences.keys())

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]

corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
paraphrases = util.paraphrase_mining(model, corpus)

#queries = ['Hotel close to Central Park',
#           'Hotel with breakfast'
#           ]

# Query sentences:
userinput = st.text_input('What kind of hotel are you looking for?')
queries = [str(userinput)]
query_embeddings = embedder.encode(queries,show_progress_bar=True)

from sentence_transformers import SentenceTransformer, util
import torch
embedder = SentenceTransformer('all-MiniLM-L6-v2')

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

def plot_cloud(wordcloud):
    # Set figure size
    #st.pyplot.figure(figsize=(40, 30))
    # Display image
    st.pyplot(wordcloud) 
    # No axis details
    #st.pyplot.axis("off");

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    st.write("\n\n======================\n\n")
    st.write("Query:", query)
    st.write("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        st.write("(Score: {:.4f})".format(score))
        row_dict = df.loc[df['all_review']== corpus[idx]]
        st.write("paper_id:  " , row_dict['hotel_name'] , "\n")
        wordcloud = WordCloud(width= 3000, height = 2000, random_state=1, background_color='salmon', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate(str(corpus[idx]))
        #plot_cloud(wordcloud)
        st.pyplot(wordcloud, interpolation='bilinear')
        #st.pyplot.axis("off")
        #st.pyplot.show()
        st.write()
