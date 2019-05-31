
# coding: utf-8

# # Analysis on whole corpus

# 1. LDA
# 2. Word2Vec
# 3. KMeans

# In[1]:

import nltk
from nltk import FreqDist
from nltk.collocations import *
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from bokeh.io import output_notebook
from bokeh.plotting import show, figure
from textblob import TextBlob
get_ipython().magic('matplotlib inline')


# In[33]:

path = "C:/Users/Sen/Downloads/CUS 635 (Web data mining)/Project/Data/Combined/"
prefix = os.listdir(path)
#print(prefix)


# Functions for normalization

# In[34]:

#Normalization
def remove_utf(text):
    return re.sub(r'[^\x00-\x7f]',r' ',text)

def remove_punctuation(corpus):
    punctuations = ".,\"-\\/#!?$%\^&\*;:{}=\-_'~()"    
    filtered_corpus = [token for token in corpus if (not token in punctuations)]
    return filtered_corpus

def apply_stopwording(corpus, min_len):
    filtered_corpus = [token for token in corpus if (not token in stopwords.words('english') and len(token)>min_len)]
    return filtered_corpus

def removeAbb(x):
    lst = {'Dx':'diagnosed' ,
           'Rx':'prescription',
           'OTC':'Over The Counter',
           'DFL':'Drug Fact Label',
           'AUT':'Application Under Test'}
    for i in x:
        if i in lst:
            i = lst[i]
    return x

def apply_lemmatization(corpus):
    lemmatizer = nltk.WordNetLemmatizer()
    normalized_corpus = [lemmatizer.lemmatize(token) for token in corpus]
    return normalized_corpus


# Removing Punctuations & Stopwording

# In[35]:

#Read the dataset
dataset={} #nltk text from tokens
dataset_raw = {} 
allFeatures=set()
tot_articles = 0
articles_count={}
raw_corpus = {} #used in sumerization
dataset2= set()
dataset3=[]
N={} # Number of articles in each corpus

for i,_ in enumerate(prefix):
    fileName = path + prefix[i]
    f=open(fileName,'r',encoding="utf8")
    text = ''
    text_raw = '' 
    
    lines = f.readlines()
    #print(_,'OK') #load test
    tot_articles+=len(lines)
    articles_count[str(_)] = len(lines)
    dataset_raw[str(_)] = list(map(lambda line: line.lower(), lines))

    for line in lines:
        dataset2.add(line.lower())
        dataset3.append(remove_utf(line.lower()))
        text+=line.replace('\n',' ').lower()
        text_raw = line.lower()
    f.close
    N[str(_)]=len(lines)
    
    tokens = nltk.word_tokenize(text)
    dataset[str(_)] = nltk.Text(tokens)
    raw_corpus[_] = text


# In[5]:

#Preprocessing
dataset_clean={} #dict of tokens

for i in dataset:
    #print ('Processing %s' % str(i))
    dataset_clean[i] = apply_lemmatization(removeAbb(apply_stopwording(remove_punctuation(dataset[i]), 3)))
    #print (dataset_clean[i])


#create a nltk.Text dict with clened dataset
clean_text = {}
for i in dataset_clean:
    clean_text[i] = nltk.Text(dataset_clean[i])

clean_text2 = ""
for i in raw_corpus:
    clean_text2 = raw_corpus[i]


# In[6]:

from nltk.corpus import wordnet
tokens = []

def get_lemma(token):
    #Return the morphological variant of this word
    lemma = wordnet.morphy(token)

    if lemma is None:
        return token
    else:
        return lemma

lemmas = [get_lemma(token) for token in tokens]
stop_words= set(nltk.corpus.stopwords.words('english'))
tokens_clean = [token for token in lemmas if (len(token)>4 and token not in stop_words)]

def clean_tokens(text):
    tokens_raw = nltk.word_tokenize(text)
    tokens = []
    for token in tokens_raw:
        if (token == " " or token.startswith('http') or token.startswith('@')):
            continue
        else:
            tokens.append(token)
    
    lemmas = [get_lemma(token) for token in tokens]
    tokens_clean = [token for token in lemmas if (len(token)>4 and token not in stop_words)]
    return tokens_clean

tokens = [clean_tokens(token) for token in dataset3]


# # LDA

# To discover the main topics, the focus group is talking about, LDA (Latent Dirichlet Allocation) is used which gives the best results on the focus group data using four clusters of words. The main topic in the corpus is a study about prostate cancer symptoms and the second biggest topic is consumer product labels that are related to prostate cancer. Some of the important topics are shown below.

# In[7]:

from gensim import corpora
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(tweet) for tweet in tokens]

import gensim
k=4
iterations = 20
topic_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=k, id2word = dictionary, passes = iterations)
topics = topic_model.print_topics(num_words = 4)
for topic in topics:
    print (topic)


# In[8]:

import pyLDAvis.gensim
lda_vis = pyLDAvis.gensim.prepare(topic_model,corpus,dictionary,sort_topics=False)
pyLDAvis.display(lda_vis)


# # Word2Vec

# Word2vec comes in action to discover words in the corpus that share common context which produces vector space of several hundreds of dimensions, to reduce the dimensionality TSNE is used to word vectors to transform the 64-dimensional space into a 2D space and by using the bokeh.iobokeh.io library we can visualize the output.

# In[36]:

cleantokens=[]
for i in dataset_clean.values():
    cleantokens+=i
#print (cleantokens[0:10])


# In[37]:

w2v_model = Word2Vec([cleantokens],size=32, sg=1, window = 5, min_count=3, seed = 20, workers=2)
print (len(w2v_model.wv.vocab))


# In[11]:

w2v_model.most_similar('patient')


# In[12]:

#Retrieving the vocabulary from the 64-dimensional space
X_32D=w2v_model[w2v_model.wv.vocab]
# Transform the data and load up a Panda dataframe
tSNE = TSNE(n_components=2, n_iter=1000)
X_2D = tSNE.fit_transform(X_32D)
x2D_df = pd.DataFrame(X_2D, columns=['x','y'])
x2D_df['word'] = w2v_model.wv.vocab.keys()
x2D_df.head(10)


# In[38]:

# Configure the notebook to generate graph in a cell
# Always call this method before any visualization
output_notebook()

# Extract a sample. If you have a powerful computer you can display all 17,000
plot = figure(plot_width=800, plot_height=800)
_ = plot.text(x=x2D_df.x, y=x2D_df.y, text=x2D_df.word)
show(plot)


# In[15]:

print(w2v_model.most_similar(positive=['patient','risk','cancer']))


# # KMeans

# In[16]:

def apply_stemming(text):
    stemmer = nltk.PorterStemmer()
    normalized_text = [stemmer.stem(token) for token in text]
    return normalized_text


# In[31]:

clean_tokens2= []
tokenized_token = []
index = 1
for token in dataset2:
    index+=1
    tokens = apply_stopwording(remove_punctuation(nltk.Text(nltk.word_tokenize(token))), 3)
    clean_text = apply_stemming(tokens)
    #print ('[%s] - %s' % (index, clean_text))
    clean_tokens2.append(clean_text)
    tokenized_token.append(tokens)


# In[32]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
documents = [str((i)) for i in clean_tokens2]
#Tf-idf matrix
tfidf_vectorizer = TfidfVectorizer(min_df=1)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
#print(documents)
#print(tfidf_matrix)
features=tfidf_vectorizer.get_feature_names()
#print(features)
#print(tfidf_matrix.shape)


# In[27]:

lemmas_list=[]
token_list=[]

lemmas_list.extend(l for lemma in clean_tokens2 for l in lemma)
token_list.extend(t for token in tokenized_token for t in token)

#token_dataframe = pd.DataFrame({'terms': token_list}, index = lemmas_list)
#print(token_dataframe.head(10))


# In[30]:

from sklearn.cluster import KMeans
k = 6
k_means = KMeans(n_clusters=k)
get_ipython().magic('time k_means.fit(tfidf_matrix)')
clusters = k_means.labels_.tolist()
tokens_space = {'term':documents, 'cluster':clusters}
kmean_dataframe = pd.DataFrame(tokens_space,index=[clusters], columns =['term','cluster'])

n=5

print ('Top %s terms within clusters' % n)
print()

sorted_centroids = k_means.cluster_centers_.argsort()[:, ::-1]
sent =""

for cluster_number in range(k):
    token_string = ''
    
    for ind in sorted_centroids[cluster_number, :n]:
        token_string = token_string + token_dataframe.loc[features[ind].split(' ')].values.tolist()[0][0] + ', '
    
    sent= sent + token_string
    print(token_string)
    print()
    print(type(token_string))
    print("Cluster %d: %s" % (cluster_number, token_string))


# In[ ]:



