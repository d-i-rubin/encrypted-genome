from sklearn.metrics import confusion_matrix
import pandas as pd
from seal import *
import numpy as np
import pickle

diags = np.load('diags.dump', allow_pickle=True)
batch_diags = np.load('batch_diags.dump', allow_pickle=True)
sketches = pickle.load(open('sketches.dump', 'rb'))
test_sketches = pickle.load(open('test_sketches.dump', 'rb'))
anchor_indices = pickle.load(open('anchor_indices.dump', 'rb'))
data_df = pd.read_pickle('data_df.dump')
test_samples = pd.read_pickle('test_samples.dump')

#DATA OWNER preprocessing
#Model owner sends data owner sketches of the anchor samples 
#Data owner computes vector of distances to each of the 12 anchors
#for each test sample.
#These vectors will be hidden by the encryption.
import time

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


#MODEL OWNER offline
#Important polynomial libraries
import numpy.polynomial.chebyshev as C
from numpy.polynomial import Polynomial as P

#MODEL OWNER offline
#This defines a degree 9 polynomial which is a good L-infinity approximation
#to a threshold function at .5 on the interval [-3,3]
#This has shown the best performance
approx = C.Chebyshev.interpolate(lambda x: .5*np.tanh(10*(x-.5))+.5,9,domain=[-3,3]).convert(kind=P)


#A plot of approx
#import matplotlib.pyplot as plt

#x = np.linspace(-3.25,3.25,100)
#ax = plt.plot(x, approx(x))

#ax = plt.plot(x,np.heaviside(x-.5,.5))
#plt.show()


approx9 = C.Chebyshev.interpolate(
    lambda x: .5*np.tanh(20*(x-.6))+.5,15,domain=[-3,3]).convert(kind=P)

#x = np.linspace(-3,3,100)
#ax = plt.plot(x, approx9(x))

#ax = plt.plot(x,np.heaviside(x-.6,.5))
#plt.show()


# In[40]:


approx15 = C.Chebyshev.interpolate(
    lambda x: .5*np.tanh(20*(x-.5))+.5,15,domain=[-3,3]).convert(kind=P)

#x = np.linspace(-3,3,100)
#ax = plt.plot(x, approx15(x))

#ax = plt.plot(x,np.heaviside(x-.5,.5))
#plt.show()


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


#ENCRYPTION, EVALUATION, DECRYPTION

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

#DATA OWNER postprocessing of results
#The first 4 entries of scores are the correct entries of the matrix-vector product. 
final_junk_removed = []
for j in range(1000):
    final_junk_removed.append(final_scores[j // 341][24*(j%341):24*(j%341)+4])
    
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

pred = conv_to_pred(final_junk_removed)

test_labels = data_df.loc[list(test_samples),0]

print(confusion_matrix(pred,test_labels))
