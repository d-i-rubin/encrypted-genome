#!/usr/bin/env python
# coding: utf-8

description = """
Step 4: Decryption and Results
------------------------------

"""

import sys
import argparse


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        description=description,
        formatter_class=CustomFormatter)
    parser.add_argument(
        "--data_owner_labels_file",
        required=False,
        type=str,
        default='1/public/data_owner_labels',
        help="Input data owners label file.")
    parser.add_argument(
        "--parms_file",
        required=False,
        type=str,
        default="2/public/IDASH_parms",
        help="Input parms file.")
    parser.add_argument(
        "--public_key_file",
        required=False,
        type=str,
        default="2/public/IDASH_pubkey",
        help="Input public key file.")
    parser.add_argument(
        "--secret_key_file",
        required=False,
        type=str,
        default='2/private/IDASH_secretkey',
        help="Input public key file.")
    parser.add_argument(
        "--ct_results_files",
        required=False,
        type=str,
        default='3/public/IDASH_ct_results_%s',
        help="Input ct results files.")
    parser.add_argument(
        "--scale_file",
        required=False,
        type=str,
        default="2/public/IDASH_scale",
        help="Input scale file.")
    args = parser.parse_args(argv[1:])
    return args


args = cli(sys.argv)
data_owner_labels_file = args.data_owner_labels_file
parms_file = args.parms_file
public_key_file = args.public_key_file
secret_key_file = args.secret_key_file
ct_results_files = args.ct_results_files
scale_file = args.scale_file


# In[1]:


#Import necessary libraries

import numpy as np
import pandas as pd
import seal
from seal import *
import time
import pickle


# In[2]:


#Data owner reloads encryption context

parms = EncryptionParameters(scheme_type.ckks)
parms.load(parms_file)

context = SEALContext(parms)
ckks_encoder = CKKSEncoder(context)


# In[3]:


keygen = KeyGenerator(context)
secret_key = keygen.secret_key()
secret_key.load(context, secret_key_file)

public_key = keygen.create_public_key()
public_key.load(context, public_key_file)

scale = pickle.load(open(scale_file,'rb'))


# In[4]:


encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)


# In[5]:


#Initialize and load ciphertexts from Model Owner, Step 3

results = []
for i in range(3):
    pt_init = ckks_encoder.encode(0.,scale)
    ct_init = encryptor.encrypt(pt_init)
    ct_init.load(context, ct_results_files % i)
    results.append(ct_init)


# In[6]:


decryptor = Decryptor(context, secret_key)


# In[7]:


#Decrypt and decode results

final_scores = []
for i in range(3):
    pt_final = decryptor.decrypt(results[i])
    final_scores.append(ckks_encoder.decode(pt_final))


# In[8]:


#DATA OWNER postprocessing of results
#The first 4 entries of scores are the correct entries of the matrix-vector product. 
final_junk_removed = []
for j in range(1000):
    final_junk_removed.append(final_scores[j // 341][24*(j%341):24*(j%341)+4])
    
final_junk_removed


# In[9]:


#DATA OWNER postprocessing
#Very simple function to convert final_junk_removed to list of label predictions
#Correct conversion assumes all scores in final_junk_removed round to 0 or 1
#Could update function to change scores >1.5 to 0 and <-.5 to 1
#This version miscategorizes all 0's as belonging to the 0th category
#and if more than one category has a 1, the reported category is the sum.
def conv_to_pred(score_array):
    predictions = []
    for scores in score_array:
        predictions.append(int(np.dot(np.round(scores),np.array([0,1,2,3]))))
        
    return predictions


# In[10]:


from sklearn.metrics import classification_report,confusion_matrix


# In[11]:


test_labels = pd.read_pickle(data_owner_labels_file)


# In[12]:


pred = conv_to_pred(final_junk_removed)


# In[13]:


#21 samples classified incorrectly, for 2% error.
#Likely all error coming from application of threshold polynomial.

print(confusion_matrix(pred,test_labels))


# In[14]:


#DATA OWNER postprocessing
#Very simple function to convert final_junk_removed to list of label predictions
#Just takes index of maximum score
def conv_to_pred2(score_array):
    predictions = []
    for scores in score_array:
        predictions.append(np.argmax(scores))
        
    return predictions


# In[15]:


#This method fails for every sample in category 0
#but works for the other categories.
#Must figure out why.
#But for now use conv_to_pred

pred2 = conv_to_pred2(final_junk_removed)
print(confusion_matrix(pred2, test_labels))

