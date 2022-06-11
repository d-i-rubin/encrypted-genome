import pickle
import numpy as np
import pandas as pd
import seal
from seal import *
import sourmash as smsh
import time
import matplotlib.pyplot as plt


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


data = load_data()
data_df = pd.DataFrame(data)
labels = data_df[:][0]


base_dict = {0:'A',1:'C',2:'G',3:'T'}

def fill_out_data(data):
    #Key preprocessing step:
    #Replace all non-ACTG characters with an ACTG chosen uniformly at random.
    data_Nrand = []

    for i in range(len(data)):
        string_mod = ''
        for j in range(len(data[i][2])):
            if data[i][2][j]=='A' or data[i][2][j]=='C' or data[i][2][j]=='G' or data[i][2][j]=='T':
                string_mod += data[i][2][j]
            else:
                string_mod+= base_dict[np.random.randint(0,4)]
        data_Nrand.append([data[i][0],data[i][1],string_mod])
    return data_Nrand

data_Nrand = fill_out_data(data)

def generate_sketches(data_Nrand):
    #These are the sketch parameters that I settled on. Form sketches of all samples.
    sketches = []
    N = 5000
    K = 33

    for i in range(len(data_Nrand)):
        mh = smsh.MinHash(n=N,ksize=K)
        mh.add_sequence(data_Nrand[i][2])
        sketches.append(mh)
    return sketches

sketches = generate_sketches(data_Nrand)


sketches_df = pd.DataFrame(sketches)

#Set aside 1000 samples as test set held by data owner.
s = pd.Series(np.arange(8000))
test_samples = s.sample(n=1000, random_state = 101)

test_sketches = sketches_df.iloc[list(test_samples)]

test_indices = list(test_samples)
test_indices.sort(reverse=True)

#Hold test sketches aside from training set.
for i in range(len(test_indices)):
    sketches.pop(test_indices[i])

#Remove test labels
for i in range(len(test_indices)):
    labels.pop(test_indices[i])


def compute_jacc_sim(sketches):
    #Compute full matrix of Jaccard similarities. Takes a long time.
    jacc_sim = np.zeros((7000,7000))

    for i in range(len(sketches)):
        #print(i)
        for j in range(i+1,len(sketches)):
            jacc_sim[i,j] = round(sketches[i].jaccard(sketches[j]),4)
    return jacc_sim

jacc_sim = compute_jacc_sim(sketches)
# XXX - Speed things up for dev work by doing the following instead of
# re-running the function. Basically you first run it once and then save the
# data like this:
#
# jacc_sim.dump('logmodel_intercept.dump')
#
# and then you load it in like this:
#
# jacc_sim = np.load('jacc_sim.dump', allow_pickle=True)

def generate_dist_adj(jacc_sim):
    #Turn Jaccard similarities into matrix of distances.
    dist_adj = np.zeros((7000,7000))

    for i in range(7000):
        #print(i)
        for j in range(i+1,7000):
            dist_adj[i,j] = -np.log(2*jacc_sim[i,j])+np.log(1+jacc_sim[i,j])
            dist_adj[j,i] = dist_adj[i,j]
    return dist_adj

dist_adj = generate_dist_adj(jacc_sim)
dist_adj_df = pd.DataFrame(dist_adj)


from sklearn.model_selection import train_test_split


#Model is based on distances to 12 randomly chosen "anchor" samples
s = pd.Series(np.arange(7000))
anchors = s.sample(n=12,random_state=5)
anchor_indices = list(anchors)


#Split Model Owner's samples into his own training and test set for validation
X_train, X_test, y_train, y_test = train_test_split(
    dist_adj_df[anchor_indices], np.ravel(labels), test_size=0.15, random_state=5)


from sklearn.linear_model import LogisticRegression


#MODEL OWNER (TRAINING)
#Fit a logistic regression model based on distances to anchors.
logmodel = LogisticRegression(fit_intercept=False)
logmodel.fit(X_train,y_train)


predictions = logmodel.predict(X_test)


#The 4*12 matrix of coefficients of the model. 
#This is the IP the Model Owner wishes to protect
#logmodel.coef_


from sklearn.metrics import classification_report,confusion_matrix


#Validate the model on a test set (not the Data Owner's test set)
print(confusion_matrix(y_test,predictions))

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


#MODEL OWNER (offline phase)
#Convert matrix of coefficients into appropriate form for mat-vec product under HE
#And batch the coefficients to apply to multiple data samples at once
diags = plain_diagonals(logmodel.coef_)

batch_diags = np.zeros((4,8192))
for i in range(0,4):
    for j in range(0,341):
        batch_diags[i][24*j:24*j+12] = diags[i]
        
batch_diags[1][:48]

logmodel.classes_.dump('logmodel_classes.dump')
logmodel.intercept_.dump('logmodel_intercept.dump')
logmodel.coef_.dump('logmodel_coef.dump')
batch_diags.dump('batch_diags.dump')
pickle.dump(test_sketches, open('test_sketches.dump','wb'))
pickle.dump(sketches, open('sketches.dump','wb'))
pickle.dump(anchor_indices, open('anchor_indices.dump', 'wb'))
diags.dump('diags.dump')
data_df.to_pickle('data_df.dump')
test_samples.to_pickle('test_samples.dump')
