#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


#Import necessary libraries

import numpy as np
import pandas as pd
import seal
from seal import *
import sourmash as smsh
import time
import matplotlib.pyplot as plt


# In[3]:


import os
import shutil

try:
    shutil.rmtree('1')
except FileNotFoundError:
    pass
os.makedirs('1/public')
os.makedirs('1/private')


# In[4]:


#Load 8000 labelled samples, comprising training and test data
def load_data():
    with open("Challenge.fa", "r") as f:
        data = f.readlines()

    labels = []
    sequences = []
    lengths = []
    for k in range(len(data)):
        if k % 2 == 0:
            labels.append(data[k])
        else:
            seq = data[k].strip()
            lengths.append(len(seq))
            sequences.append(seq)

    # uniformize lengths by filling in with N's
    #max_length = max(lengths)
    #for i in range(len(sequences)):
        #padding_size = max_length - len(sequences[i])
        #for j in range(padding_size):
            #sequences[i] += "N"


    types = [">B.1.526", ">B.1.1.7", ">B.1.427", ">P.1"]

    dataframe = []

    for i in range(len(labels)):
        entry = []
        # 2021/08/02: re-replaced use of match-case (Python 3.10) for backwards compatibility
        for j in range(len(types)):
            if labels[i].startswith(types[j]):
                entry.append(j)
                virus_number = labels[i].split("_")[1].strip()
                entry.append(virus_number)
                entry.append(sequences[i])
                break

            if j == 3:
                raise "Bad entry"

        dataframe.append(entry)

    return dataframe


# In[5]:


data = load_data()


# In[6]:


data_df = pd.DataFrame(data)
labels = data_df[:][0]


# In[7]:


base_dict = {0:'A',1:'C',2:'G',3:'T'}


# In[8]:


#Key preprocessing step:
#Replace all non-ACTG characters with an ACTG chosen uniformly at random.
start = time.time()
data_Nrand = []

for i in range(len(data)):
    string_mod = ''
    for j in range(len(data[i][2])):
        if data[i][2][j]=='A' or data[i][2][j]=='C' or data[i][2][j]=='G' or data[i][2][j]=='T':
            string_mod += data[i][2][j]
        else:
            string_mod+= base_dict[np.random.randint(0,4)]
    data_Nrand.append([data[i][0],data[i][1],string_mod])
    
end = time.time()
print(f'Time to Replace unknowns: {(end-start):.3f}s')


# In[9]:


#These are the sketch parameters that I settled on. Form sketches of all samples.
start = time.time()
sketches = []
N = 5000
K = 33

for i in range(len(data_Nrand)):
    mh = smsh.MinHash(n=N,ksize=K)
    mh.add_sequence(data_Nrand[i][2])
    sketches.append(mh)
    
end = time.time()
print(f'Time to form sketches: {(end-start):.3f}s')


# In[10]:


#Set aside 1000 samples as test set held by data owner.
s= pd.Series(np.arange(8000))
test_samples = s.sample(n=1000, random_state = 101)
test_samples


# In[11]:


sketches_df = pd.DataFrame(sketches)
test_sketches = sketches_df.iloc[list(test_samples)]


# In[12]:


test_labels = labels[list(test_samples)]
test_labels


# In[13]:


#Save test labels for Data Owner to access in Step 4.
#In real situation these would not be accessible to Model Owner.

test_labels.to_pickle('1/public/data_owner_labels')


# In[14]:


test_indices = list(test_samples)
test_indices.sort(reverse=True)


# In[15]:


#Hold test sketches aside from training set.
for i in range(len(test_indices)):
    sketches.pop(test_indices[i])


# In[16]:


#Remove test labels
for i in range(len(test_indices)):
    labels.pop(test_indices[i])


# In[17]:


#MODEL OWNER (TRAINING)
#Compute full matrix of Jaccard similarities. Takes a long time.
start = time.time()
jacc_sim = np.zeros((7000,7000))

for i in range(len(sketches)):
    #print(i)
    for j in range(i+1,len(sketches)):
        jacc_sim[i,j] = round(sketches[i].jaccard(sketches[j]),4)
        
end = time.time()
print(f'Time to compute similarities between all training sketches: {(end-start):.3f}s')


# In[18]:


#Turn Jaccard similarities into matrix of distances.
start = time.time()
dist_adj = np.zeros((7000,7000))

for i in range(7000):
    #print(i)
    for j in range(i+1,7000):
        dist_adj[i,j] = -np.log(2*jacc_sim[i,j])+np.log(1+jacc_sim[i,j])
        dist_adj[j,i] = dist_adj[i,j]
        
end = time.time()
print(f'Time to compute training distances: {(end-start):.3f}s')


# In[19]:


dist_adj_df = pd.DataFrame(dist_adj)


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


#Model is based on distances to 12 randomly chosen "anchor" samples
s = pd.Series(np.arange(7000))
anchors = s.sample(n=12,random_state=5)


# In[22]:


anchor_indices = list(anchors)


# In[23]:


#Split Model Owner's samples into his own training and test set for validation
X_train, X_test, y_train, y_test = train_test_split(dist_adj_df[anchor_indices], np.ravel(labels), test_size=0.15, random_state=5)


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


#MODEL OWNER (TRAINING)
#Fit a logistic regression model based on distances to anchors.
logmodel = LogisticRegression(fit_intercept=False)
logmodel.fit(X_train,y_train)


# In[26]:


predictions = logmodel.predict(X_test)


# In[27]:


#The 4*12 matrix of coefficients of the model. 
#This is the IP the Model Owner wishes to protect
logmodel.coef_


# In[28]:


from sklearn.metrics import classification_report,confusion_matrix


# In[29]:


#Validate the model on a test set (not the Data Owner's test set)
print(confusion_matrix(y_test,predictions))


# In[30]:


#Save model data for Model Owner's use in Step 3

logmodel.classes_.dump('1/private/logmodel_classes.dump')
logmodel.intercept_.dump('1/private/logmodel_intercept.dump')
logmodel.coef_.dump('1/private/logmodel_coef.dump')


#Save test sketches for Data Owner in Step 2
#In real situation, Data Owner would hold these from the start

pickle.dump(test_sketches, open('1/public/test_sketches.dump','wb'))

#Data below isn't used again.

#pickle.dump(sketches, open('sketches.dump','wb'))
#pickle.dump(anchor_indices, open('anchor_indices.dump', 'wb'))
#data_df.to_pickle('data_df.dump')
#test_samples.to_pickle('test_samples.dump')


# In[31]:


anchor_sketches = sketches_df.iloc[anchor_indices]
anchor_sketches


# In[32]:


test_sketches


# In[33]:


#Save anchor sketches to send to Data Owner in Step 2

pickle.dump(anchor_sketches, open('1/public/anchor_sketches.dump','wb'))

