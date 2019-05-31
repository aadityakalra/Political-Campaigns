
# coding: utf-8

# # FocusGroupBot
# ![image.png](attachment:image.png)

# In[1]:

#pip install chatterbot in cmd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import pandas as pd


# In[2]:

chatbot = ChatBot('FocusGroupBot', logic_adapters=[
        {
            "import_path": "chatterbot.logic.BestMatch",
            "statement_comparison_function": "chatterbot.comparisons.levenshtein_distance",
            "response_selection_method": "chatterbot.response_selection.get_first_response"
        }])


# # Training the chatbot

# In[3]:

chatbot.set_trainer(ListTrainer)


# In[4]:

f = open('./trainbot.txt','r',encoding="utf8").readlines()
chatbot.train(f)
#g = open('./definitions.txt','r',encoding="utf8").readlines()
#chatbot.train(g)


# Question the bot trained on based on corpus and summary data.

# In[5]:

Q = []
for i,ch in enumerate(f):
    x = ch.split(",")
    #Q['test'] = x[0]
    Q.append(x[0])
    
print(Q) 


# Q = []
# for i,ch in enumerate(train):
#     if i/2 != 0:
#         Q.append(ch)

# Some random questions

# In[6]:

Q_test = [
    "who are alpha blocker given to?",
    "What's in alpha blockers",
    "concerns of pre existing condition taking tamsulosin hydrochloride?",
    "will the FDA be concerned about patients with preexcisting condition taking tamsulosin hydrochloride?",
    "the risk of tamsulosin hydrochloride?",
    "should patients be pre screened prior to treatment with tamsulosin hydrochloride?",
    "Is a dipstick necessary?"
]


# Test is out, ask it random stuff...
# "definition of *" for definitions

# In[7]:

question = "Does company have any information regarding tamsulosin?"
print(chatbot.get_response(question))


# In[8]:

question = "What is the limitations of OTC label?"
print(chatbot.get_response(question))


# In[9]:

question = "What is the limitations of OTC label?"
print(chatbot.get_response(question))


# In[10]:

for i in Q_test:
    print('User:', i)
    print('Bot:',chatbot.get_response(i))
    print()


# In[ ]:



