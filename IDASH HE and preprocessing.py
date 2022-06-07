#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries

import numpy as np
import pandas as pd
import seal
from seal import *
import sourmash as smsh
import time
import matplotlib.pyplot as plt


# In[2]:


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


# In[3]:


data = load_data()


# In[4]:


data[4321]


# In[5]:


data_df = pd.DataFrame(data)
labels = data_df[:][0]
labels


# In[6]:


base_dict = {0:'A',1:'C',2:'G',3:'T'}
base_dict


# In[7]:


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
    
data_Nrand[4444]


# In[8]:


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


# In[9]:


#Set aside 1000 samples as test set held by data owner.
s= pd.Series(np.arange(8000))
test_samples = s.sample(n=1000, random_state = 101)
test_samples


# In[10]:


sketches_df = pd.DataFrame(sketches)
sketches_df.head()


# In[11]:


test_sketches = sketches_df.iloc[list(test_samples)]
test_sketches.head()


# In[12]:


test_indices = list(test_samples)
test_indices.sort(reverse=True)
test_indices


# In[13]:


#Hold test sketches aside from training set.
for i in range(len(test_indices)):
    sketches.pop(test_indices[i])


# In[14]:


len(sketches)


# In[15]:


#Remove test labels
for i in range(len(test_indices)):
    labels.pop(test_indices[i])


# In[16]:


labels


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
anchors


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


predictions[:10]


# In[28]:


#The 4*12 matrix of coefficients of the model. 
#This is the IP the Model Owner wishes to protect
logmodel.coef_


# In[29]:


from sklearn.metrics import classification_report,confusion_matrix


# In[30]:


#Validate the model on a test set (not the Data Owner's test set)
print(confusion_matrix(y_test,predictions))


# In[31]:


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


# In[32]:


#MODEL OWNER (offline phase)
#Convert matrix of coefficients into appropriate form for mat-vec product under HE
#And batch the coefficients to apply to multiple data samples at once
diags = plain_diagonals(logmodel.coef_)

batch_diags = np.zeros((4,8192))
for i in range(0,4):
    for j in range(0,341):
        batch_diags[i][24*j:24*j+12] = diags[i]
        
batch_diags[1][:48]


# In[ ]:





# In[33]:


#DATA OWNER preprocessing
#Model owner sends data owner sketches of the anchor samples 
#Data owner computes vector of distances to each of the 12 anchors
#for each test sample.
#These vectors will be hidden by the encryption.
import time

start = time.time()
jacc_sim = np.zeros((1000,12))

i=0
for sketch in test_sketches[0]:
    for j in range(0,12):
        jacc_sim[i,j] = round(sketch.jaccard(sketches[anchor_indices[j]]),4)
    i+=1
        
dist_data = np.zeros((1000,12))

for i in range(1000):
    for j in range(12):
        dist_data[i,j] = -np.log(2*jacc_sim[i,j])+np.log(1+jacc_sim[i,j])
        
end = time.time()
print(f'Time to Preprocess Data: {(end-start):.3f}s')


# In[34]:


dist_data


# In[35]:


#DATA OWNER preprocessing
#Batches 341 samples into a single plaintext (8192 slots / (12 + 12 empty))
#1000 samples are placed into 3 large vectors
#batch_data[2] has extra 0's at the end
batch_data = np.zeros((3,8192))
for i in range(3):
    for j in range(341):
        if 341*i+j < 1000:
            batch_data[i][24*j:24*j+12] = dist_data[341*i+j]
    
    
batch_data[2][-600:]


# In[36]:


#MODEL OWNER offline
#Important polynomial libraries
import numpy.polynomial.chebyshev as C
from numpy.polynomial import Polynomial as P


# In[37]:


#MODEL OWNER offline
#This defines a degree 9 polynomial which is a good L-infinity approximation
#to a threshold function at .5 on the interval [-3,3]
#This has shown the best performance
approx = C.Chebyshev.interpolate(lambda x: .5*np.tanh(10*(x-.5))+.5,9,domain=[-3,3]).convert(kind=P)


# In[38]:


#A plot of approx
import matplotlib.pyplot as plt

x = np.linspace(-3.25,3.25,100)
ax = plt.plot(x, approx(x))

ax = plt.plot(x,np.heaviside(x-.5,.5))
plt.show()


# In[39]:


approx9 = C.Chebyshev.interpolate(lambda x: .5*np.tanh(20*(x-.6))+.5,15,domain=[-3,3]).convert(kind=P)

x = np.linspace(-3,3,100)
ax = plt.plot(x, approx9(x))

ax = plt.plot(x,np.heaviside(x-.6,.5))
plt.show()


# In[40]:


approx15 = C.Chebyshev.interpolate(lambda x: .5*np.tanh(20*(x-.5))+.5,15,domain=[-3,3]).convert(kind=P)

x = np.linspace(-3,3,100)
ax = plt.plot(x, approx15(x))

ax = plt.plot(x,np.heaviside(x-.5,.5))
plt.show()


# In[41]:


#DATA OWNER
#Set the parameters of the encryption context.

parms = EncryptionParameters(scheme_type.ckks)
poly_modulus_degree = 16384
parms.set_poly_modulus_degree(poly_modulus_degree)
parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, [60, 40, 40, 40, 40, 40, 60]))
#320-bit coeff modulus Q. 
#From SEAL manual, security cutoffs for N=16384 are 300 bits for 192-bit security, 438 bits for 128-bit security.
scale = 2.0**40
context = SEALContext(parms)
#print_parameters(context)

print(CoeffModulus.MaxBitCount(poly_modulus_degree))

ckks_encoder = CKKSEncoder(context)
slot_count = ckks_encoder.slot_count()
print(f'Number of slots: {slot_count}')

keygen = KeyGenerator(context)
public_key = keygen.create_public_key()
secret_key = keygen.secret_key()
galois_keys = keygen.create_galois_keys()
relin_keys = keygen.create_relin_keys()

encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
#decryptor = Decryptor(context, secret_key)


# In[42]:


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


# In[43]:


def threshold_poly(ct_score,poly_approx,evaluator,relin_keys):
    
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


# In[44]:


#ENCRYPTION, EVALUATION, DECRYPTION

start = time.time()


    
#DATA OWNER
#Encode and encrypt data owner's distance vector
pt_data = []
ct_data = []

for i in range(3):
    pt_data.append(ckks_encoder.encode(batch_data[i],scale))
    ct_data.append(encryptor.encrypt(pt_data[i]))

#MODEL OWNER
#HERE MODEL OWNER MUST RECEIVE ct_data, context, evaluator, galois_keys, and relin_keys
#Encode plaintext vectors
pt_model_diags = []
for i in range(0,diags.shape[0]):
    pt_model_diags.append(ckks_encoder.encode(batch_diags[i],scale))

#Model owner runs the encrypted calculation
final = []
for i in range(3):
    #Linear mat-vec product between model matrix and data
    result = compute_logmodel_linear(ct_data[i],pt_model_diags,evaluator, galois_keys)

    #Apply polynomial to approximate 0/1 classification vector
    final.append(threshold_poly(result,approx.coef,evaluator,relin_keys))

#evaluator.rescale_to_next_inplace(final)

#DATA OWNER
#Data owner receives final from Model Owner
pt_final = []
final_scores = []
decryptor = Decryptor(context, secret_key)
for i in range(3):
    pt_final.append(decryptor.decrypt(final[i]))
    final_scores.append(ckks_encoder.decode(pt_final[i]))

#pt_result = decryptor.decrypt(result)
#raw_scores = ckks_encoder.decode(pt_result)

#print(raw_scores)
end = time.time()
print(f'Time to Encrypt, Evaluate, and Decrypt: {(end-start):.3f}s')




# In[45]:


#DATA OWNER postprocessing of results
#The first 4 entries of scores are the correct entries of the matrix-vector product. 
final_junk_removed = []
for j in range(1000):
    final_junk_removed.append(final_scores[j // 341][24*(j%341):24*(j%341)+4])
    
final_junk_removed


# In[46]:


np.round(final_junk_removed)[:10]


# In[47]:


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


# In[48]:


pred = conv_to_pred(final_junk_removed)


# In[49]:


test_labels = data_df.loc[list(test_samples),0]


# In[50]:


print(confusion_matrix(pred,test_labels))


# In[ ]:




