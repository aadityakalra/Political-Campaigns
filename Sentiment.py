
# coding: utf-8

# # Sentiment Analysis

# 1. Textblob
# 2. Sentiword 3.0

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


# In[12]:

path = "C:/Users/Sen/Downloads/CUS 635 (Web data mining)/Project/Data/Combined/"
prefix = os.listdir(path)
print(prefix)


# In[13]:

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


# In[14]:

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
    


# # Textblob

# Subjectivity of the discussion topics for each question was generated through TextBlob. Subjectivity refers to being influenced by private opinions and beliefs, while objectivity refers to measurable facts generally agreed upon societies; 0 is objective and 1 is subjective. Product. Table below shows the key concepts for each question and their corresponding sentiment scores.

# In[92]:

#textblobdf = pd.DataFrame()
for i in prefix: 
    tblog = TextBlob(str(dataset_raw[i]))
    #print(i)
    print (i,"->",tblog.sentiment)
    #textblobdf[i] = tblog.sentiment


# # SentiWordNet3.0 

# The analysis was conducted using the SentiWordNet3.0 Lexicon Dictionary for each week and question. The original lexicon was manually enhanced to better fit the specificity of the medical dataset by manipulating certain words’ scores. For instance, ‘risk’ and ‘cancer’ in all contexts were changed to negative 0.9, while words that may suggest benefits of the treatment were assigned more positive scores. The final score is a weighted average for all words in each question; with score above 1.5 sentiment is positive, score below 1.5 is negative, and scores in between indicate a neutral attitude. 

# In[16]:

#path = 'C:/Users/ziadm/Downloads/Web Data mining/'
path2 = "C:/Users/Sen/Downloads/CUS 635 (Web data mining)/Project/"
def loadSentiWordNet(lfile):
    lf = open(lfile)
    lines=lf.readlines()
    lf.close
    lexicon = {}
    for line in lines:
        info = line.split("\t")
        try:
            p_score = float(info[2])
            n_score = float(info[3]) * -1.0
            words = info[4].split(" ")
            for word in words:
                if "#" in word:
                    term = word.split("#")
                    lexicon[term[0]]= p_score + n_score
        except:
            pass
    return lexicon

lexicon_dictionary = "newsenti.txt"
lex_fileName=path2+lexicon_dictionary
lexicon_dictionary = loadSentiWordNet(lex_fileName)


# In[17]:

testing=[]
tokensnew = nltk.word_tokenize(clean_text2)
cleaned_tokens2 = apply_stopwording(tokensnew,3)
testing.append(cleaned_tokens2)
'''
for i in testing:
    print(len(i))
    for x in range(len(i)):
        print(x)
'''
keys = lexicon_dictionary.keys()
#print(lexicon_dictionary["concern"])
score = 0.0
keys = lexicon_dictionary.keys()
for i in testing:
    for word in range(len(i)):
        if i[word] in keys:
            score = score + lexicon_dictionary[i[word]]


# In[45]:

def scoreSentiment(testing,lex_dic):
    score = 0.0
    sentiment = "Neutral"
    keys = lex_dic.keys()
    for i in testing:
        for word in range(len(i)):
            if i[word] in keys:
                #print(i[word],lexicon_dictionary[i[word]])
                score = score + lexicon_dictionary[i[word]]   
    if score >1.5:
        sentiment = "Positive"
    elif score<-1.5:
        sentiment = "Negative"
    return(sentiment, score) 

scoreSentiment(testing,lexicon_dictionary)


# In[51]:

sentiScore = pd.DataFrame()
for i in prefix: 
    dataset_raw[i]
    testing2=[]
    #tokensnew2 = nltk.word_tokenize(str(dataset_raw[i]))
    #cleaned_tokens3 = apply_stopwording(nltk.word_tokenize(str(dataset_raw[i])),3)
    testing2.append(apply_stopwording(nltk.word_tokenize(str(dataset_raw[i])),3))
    sentiScore[i] = scoreSentiment(testing2,lexicon_dictionary)


# In[52]:

sentiScore


# In[ ]:



