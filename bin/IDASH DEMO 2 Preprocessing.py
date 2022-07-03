#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries

import numpy as np
import pandas as pd
import seal
from seal import *
import sourmash as smsh
import matplotlib.pyplot as plt
import pickle


# In[2]:


import os
import shutil

try:
    shutil.rmtree('2')
except FileNotFoundError:
    pass
os.makedirs('2/public')
os.makedirs('2/private')


# In[3]:


#Data Owner loads sketches of private data and loads anchor sketches sent by Model Owner

test_sketches = pickle.load(open('1/public/test_sketches.dump', 'rb'))
anchor_sketches = pickle.load(open('1/public/anchor_sketches.dump', 'rb'))


# In[4]:


#DATA OWNER preprocessing
#Model owner sends data owner sketches of the anchor samples 
#Data owner computes vector of distances to each of the 12 anchors
#for each test sample.
#These vectors will be hidden by the encryption.

jacc_sim = np.zeros((1000,12))

i=0
for sketch in test_sketches[0]:
    j=0
    for anchor in anchor_sketches[0]:
        jacc_sim[i,j] = round(sketch.jaccard(anchor),4)
        j+=1
    i+=1
        
dist_data = np.zeros((1000,12))

for i in range(1000):
    for j in range(12):
        dist_data[i,j] = -np.log(2*jacc_sim[i,j])+np.log(1+jacc_sim[i,j])


# In[5]:


#DATA OWNER preprocessing
#Batches 341 samples into a single plaintext (8192 slots / (12 + 12 empty))
#1000 samples are placed into 3 large vectors
#batch_data[2] has extra 0's at the end
batch_data = np.zeros((3,8192))
for i in range(3):
    for j in range(341):
        if 341*i+j < 1000:
            batch_data[i][24*j:24*j+12] = dist_data[341*i+j]


# In[6]:


#DATA OWNER
#Set the parameters of the encryption context.
#In real situation, the Model Owner would communicate the parameters poly_modulus_degree,
#list of prime sizes in coeff_modulus, and scale,
#based on the number of rescalings in their evaluation.

parms = EncryptionParameters(scheme_type.ckks)
poly_modulus_degree = 16384
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 40, 40, 40, 60]))
#320-bit coeff modulus Q. 
#From SEAL manual, security cutoffs for N=16384 are 300 bits for 192-bit security, 438 bits for 128-bit security.
scale = 2.0**40
context = SEALContext(parms)
#print_parameters(context)

ckks_encoder = CKKSEncoder(context)
slot_count = ckks_encoder.slot_count()

keygen = KeyGenerator(context)
public_key = keygen.create_public_key()
secret_key = keygen.secret_key()
galois_keys = keygen.create_galois_keys()
relin_keys = keygen.create_relin_keys()

encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)


# In[7]:


#DATA OWNER
#Encode and encrypt data owner's distance vector
pt_data = []
ct_data = []

for i in range(3):
    pt_data.append(ckks_encoder.encode(batch_data[i],scale))
    ct_data.append(encryptor.encrypt(pt_data[i]))


# In[8]:


#DATA OWNER
#Save to file: ciphertext data and encryption context,
#including parameters, public key, galois keys, and relinearization keys
#This data is sent to the Model Owner for Step 3.

ct_data[0].save('2/public/ct_0')
ct_data[1].save('2/public/ct_1')
ct_data[2].save('2/public/ct_2')



parms.save('2/public/IDASH_parms')
public_key.save('2/public/IDASH_pubkey')
galois_keys.save('2/public/IDASH_galkeys')
relin_keys.save('2/public/IDASH_relinkeys')
pickle.dump(scale, open('2/public/IDASH_scale','wb'))


# In[9]:


#Data Owner saves secret key for Step 4

secret_key.save('2/private/IDASH_secretkey')

