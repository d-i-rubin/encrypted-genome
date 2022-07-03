#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries

import numpy as np
import pandas as pd
import seal
from seal import *
import time
import pickle


# In[2]:


import os
import shutil

try:
    shutil.rmtree('3')
except FileNotFoundError:
    pass
os.makedirs('3/public')
os.makedirs('3/private')


# In[3]:


model_coef = np.load('1/private/logmodel_coef.dump', allow_pickle=True)


# In[4]:


#Assume matr is mxn, where m divides n.
#Returns list of m vectors, each of which is a diagonal of length n.

def plain_diagonals(matr):
    m = matr.shape[0]
    n=matr.shape[1]
    vecs = np.zeros((m,n))
    for i in range(m):
        temp = []
        for j in range(n):
            k = j % m
            l = (j + i) % n
            temp.append(matr[k][l])
        vecs[i] = temp
    
    return vecs


# In[5]:


#MODEL OWNER (offline phase)
#Convert matrix of coefficients into appropriate form for mat-vec product under HE
#And batch the coefficients to apply to multiple data samples at once
diags = plain_diagonals(model_coef)

batch_diags = np.zeros((4,8192))
for i in range(0,4):
    for j in range(0,341):
        batch_diags[i][24*j:24*j+12] = diags[i]
        
batch_diags[1][:48]


# In[6]:


#MODEL OWNER offline
#Important polynomial libraries
import numpy.polynomial.chebyshev as C
from numpy.polynomial import Polynomial as P


# In[7]:


#MODEL OWNER offline
#This defines a degree 9 polynomial which is a good L-infinity approximation
#to a threshold function at .5 on the interval [-3,3]
#This has shown the best performance
approx = C.Chebyshev.interpolate(lambda x: .5*np.tanh(10*(x-.5))+.5,9,domain=[-3,3]).convert(kind=P)


# In[8]:


#Model Owner initializes an encryption scheme

parms = EncryptionParameters(scheme_type.ckks)


# In[9]:


#Model owner loads data owner's parameters

parms.load('2/public/IDASH_parms')


# In[10]:


#MO initializes context with DO's parms

context = SEALContext(parms)


# In[11]:


#Check the slot count

ckks_encoder = CKKSEncoder(context)
slot_count = ckks_encoder.slot_count()


# In[12]:


#MO initializes and loads keys and scale

keygen = KeyGenerator(context)
public_key = keygen.create_public_key()
public_key.load(context, '2/public/IDASH_pubkey')


galois_keys = keygen.create_galois_keys()
galois_keys.load(context, '2/public/IDASH_galkeys')


relin_keys = keygen.create_relin_keys()
relin_keys.load(context, '2/public/IDASH_relinkeys')

scale = pickle.load(open('2/public/IDASH_scale','rb'))


# In[13]:


def compute_logmodel_linear(ct_data,pt_diag_coeff,evaluator,galois_keys):
    #Copy the entries of ct_data into the next 12 slots so that rotation works properly
    ct_rot = evaluator.rotate_vector(ct_data, -12, galois_keys)
    ct_temp = evaluator.add(ct_data, ct_rot)


    #Evaluate pt-ct matrix-vector product.
    temp_result = evaluator.multiply_plain(ct_data,pt_diag_coeff[0])

    #pt_temp_result = decryptor.decrypt(temp_result)
    #print(ckks_encoder.decode(pt_temp_result)[:14])

    for i in range(1,4):
        temp_data_vec = evaluator.rotate_vector(ct_temp,i,galois_keys)
        evaluator.multiply_plain_inplace(temp_data_vec,pt_diag_coeff[i])
        evaluator.add_inplace(temp_result,temp_data_vec)
    

    evaluator.rescale_to_next_inplace(temp_result)
    #pt_temp_result = decryptor.decrypt(temp_result)
    #print(ckks_encoder.decode(pt_temp_result)[:14])

    temp_result1 = evaluator.rotate_vector(temp_result,4,galois_keys)
    temp_result2 = evaluator.rotate_vector(temp_result,8,galois_keys)

    result = evaluator.add_many([temp_result,temp_result1,temp_result2])
    
    return result


# In[14]:


def eval_deg9(ct_score,poly_approx,evaluator,relin_keys):
    
    #ct_score is ciphertext of raw score results from linear part of the logistic regression model.
    #poly_approx is the np.array of coefficients of a degree 9 polynomial,
    #poly_approx[i] the coeff of x ** i
    
    #Hide raw scores and deliver only the category with max score.
    #Approximate max with polynomial approximation of threshold function.
    powers = []
    #const1 = [1]*slot_count
    #powers.append(encryptor.encrypt(ckks_encoder.encode(const1)))
    powers.append(ct_score)
    square = evaluator.square(ct_score)

    evaluator.relinearize_inplace(square,relin_keys)
    evaluator.rescale_to_next_inplace(square)
    powers.append(square)

    evaluator.mod_switch_to_next_inplace(powers[0])
    cube = evaluator.multiply(powers[0],powers[1])
    evaluator.relinearize_inplace(cube,relin_keys)
    evaluator.rescale_to_next_inplace(cube)
    powers.append(cube)

    sixth = evaluator.square(powers[2])
    evaluator.relinearize_inplace(sixth,relin_keys)
    evaluator.rescale_to_next_inplace(sixth)
    powers.append(sixth)

    #for power in powers:
        #print(f"Scale: {power.scale()}")

              
    #Encode coefficients of approximating polynomial in plaintext
    pt_coeff = []
    for i in range(len(poly_approx)):
        pt_coeff.append(ckks_encoder.encode(poly_approx[i],scale))

    #Evaluate polynomial in babystep-gianstep form.
    evaluator.mod_switch_to_inplace(pt_coeff[1],powers[0].parms_id())
    poly_term1 = evaluator.multiply_plain(powers[0],pt_coeff[1])
    evaluator.rescale_to_next_inplace(poly_term1)
    evaluator.mod_switch_to_inplace(pt_coeff[2],powers[1].parms_id())
    poly_term2 = evaluator.multiply_plain(powers[1],pt_coeff[2])
    evaluator.rescale_to_next_inplace(poly_term2)
    evaluator.mod_switch_to_inplace(pt_coeff[3],powers[2].parms_id())
    poly_term3 = evaluator.multiply_plain(powers[2],pt_coeff[3])
    evaluator.rescale_to_next_inplace(poly_term3)


    evaluator.mod_switch_to_inplace(poly_term1,poly_term3.parms_id())
    evaluator.mod_switch_to_inplace(poly_term2,poly_term3.parms_id())


    poly_term1.scale(poly_term3.scale())
    poly_term2.scale(poly_term3.scale())

    group1 = evaluator.add_many([poly_term1,poly_term2,poly_term3])
    evaluator.mod_switch_to_inplace(pt_coeff[0],group1.parms_id())
    pt_coeff[0].scale(group1.scale())
    evaluator.add_plain_inplace(group1,pt_coeff[0])


    evaluator.mod_switch_to_inplace(pt_coeff[4],powers[0].parms_id())
    poly_term4 = evaluator.multiply_plain(powers[0],pt_coeff[4])
    evaluator.rescale_to_next_inplace(poly_term4)
    evaluator.mod_switch_to_inplace(pt_coeff[5],powers[1].parms_id())
    poly_term5 = evaluator.multiply_plain(powers[1],pt_coeff[5])
    evaluator.rescale_to_next_inplace(poly_term5)
    evaluator.mod_switch_to_inplace(pt_coeff[6],powers[2].parms_id())
    poly_term6 = evaluator.multiply_plain(powers[2],pt_coeff[6])
    evaluator.rescale_to_next_inplace(poly_term6)

    evaluator.mod_switch_to_inplace(poly_term4,poly_term6.parms_id())
    evaluator.mod_switch_to_inplace(poly_term5,poly_term6.parms_id())


    poly_term4.scale(poly_term6.scale())
    poly_term5.scale(poly_term6.scale())

    group2 = evaluator.add_many([poly_term4,poly_term5,poly_term6])


    evaluator.mod_switch_to_inplace(pt_coeff[7],powers[0].parms_id())
    poly_term7 = evaluator.multiply_plain(powers[0],pt_coeff[7])
    evaluator.rescale_to_next_inplace(poly_term7)
    evaluator.mod_switch_to_inplace(pt_coeff[8],powers[1].parms_id())
    poly_term8 = evaluator.multiply_plain(powers[1],pt_coeff[8])
    evaluator.rescale_to_next_inplace(poly_term8)
    evaluator.mod_switch_to_inplace(pt_coeff[9],powers[2].parms_id())
    poly_term9 = evaluator.multiply_plain(powers[2],pt_coeff[9])
    evaluator.rescale_to_next_inplace(poly_term9)

    evaluator.mod_switch_to_inplace(poly_term7,poly_term9.parms_id())
    evaluator.mod_switch_to_inplace(poly_term8,poly_term9.parms_id())


    poly_term7.scale(poly_term9.scale())
    poly_term8.scale(poly_term9.scale())


    evaluator.mod_switch_to_inplace(powers[2],group2.parms_id())
    evaluator.multiply_inplace(group2,powers[2])
    evaluator.relinearize_inplace(group2,relin_keys)
    evaluator.rescale_to_next_inplace(group2)

    

    group3 = evaluator.add_many([poly_term7,poly_term8,poly_term9])
    evaluator.rescale_to_inplace(powers[3],group3.parms_id())
    evaluator.multiply_inplace(group3,powers[3])
    evaluator.relinearize_inplace(group3,relin_keys)
    evaluator.rescale_to_next_inplace(group3)

    

    evaluator.mod_switch_to_inplace(group1,group3.parms_id())
    evaluator.mod_switch_to_inplace(group2,group3.parms_id())

    
    group1.scale(group3.scale())
    group2.scale(group3.scale())

    final = evaluator.add_many([group1,group2,group3])
    
    return final


# In[15]:


#Encode batched model coefficients

pt_model_diags = []
for i in range(0,diags.shape[0]):
    pt_model_diags.append(ckks_encoder.encode(batch_diags[i],scale))


# In[16]:


encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)


# In[17]:


#Initialize and load ciphertexts from Data Owner

ct_data = []
for i in range(3):
    pt_init = ckks_encoder.encode(0.,scale)
    ct_init = encryptor.encrypt(pt_init)
    ct_init.load(context, '2/public/ct_%s' % i)
    ct_data.append(ct_init)


# In[18]:


#Model owner runs the encrypted calculation
final = []
for i in range(3):
    #Linear mat-vec product between model matrix and data
    result = compute_logmodel_linear(ct_data[i],pt_model_diags,evaluator, galois_keys)

    #Apply polynomial to approximate 0/1 classification vector
    final.append(eval_deg9(result,approx.coef,evaluator,relin_keys))


# In[19]:


#Save results to send to Data Owner in Step 4

for i in range(3):
    final[i].save('3/public/IDASH_ct_results_%s' % i)
