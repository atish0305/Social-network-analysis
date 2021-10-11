# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 10:34:18 2020

@author: atish
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from collections import defaultdict
from itertools import combinations
from datetime import timedelta
from psutil import virtual_memory
import time

start = time.time()


def time_lapse(string,T1,T2):
    """
    Prints the time spent
    """
    print("Time spent on %s: %s" % (string, str(timedelta(seconds=T2-T1))))
    
    

def create_sparse(filename):
    """
    Loads the data from the file present in the local directory and create shingles of the data and return a sparse matrix. 
    """
    S = time.time()
    data = np.load(filename)
    ## Creating a sparse matrix with rows as movies and columns as users
    mat_sparse = sparse.csr_matrix((np.ones(len(data)), (data[:,1], data[:,0])), dtype = np.int32)
    time_lapse("creating the sparse matrix",S,time.time())
    return mat_sparse


def create_signature_mat(sparse_mat,permutes,seed):
    """
    Takes input of sparse matrix, number of permutations, and the seed value in order to create a min-hashed signature matrix 
    """
    S =  time.time()
    ## Setting the seed value
    np.random.seed(seed)
    num_of_movie = sparse_mat.shape[0]
    num_of_user = sparse_mat.shape[1] 
    signature_mat = np.empty((permutes, num_of_user), dtype=np.int32)
    ## Shuffling the rows and picking the first row for each column which has a non zero value
    for i in range(permutes):
        signature_mat[i] = np.array(sparse_mat[np.random.permutation(num_of_movie)].argmax(axis=0)).ravel()
    time_lapse("creating the signature matrix",S,time.time())
    return signature_mat



def jaccardsim(CP1, CP2, sparse_mat):
    """
    Finds the Jaccard similarity between two candidate pairs/users.
    """
    ## Calculating the intersection of candidate pair 1 and candidate pair 2
    Intersection = sparse_mat[:,CP2].multiply(sparse_mat[:,CP1]).sum(axis = 0).reshape(-1,1)
    ## Calculating the total number of movies seen by candidate pair 1 and candidate pair 2
    TotCP1_CP2 = sparse_mat[:,CP2].sum(axis = 0).reshape(-1,1) + sparse_mat[:,CP1].sum(axis = 0).reshape(-1,1)
    return Intersection/(TotCP1_CP2-Intersection)




def create_hash_table(signature_mat, permutes, bucket_size):
    """
    Takes input of signature matrix, number of permutations, and bucket size in order to create a hash table of the users
    """
    rows = int(permutes/bucket_size)     
    num_of_users = signature_mat.shape[1]
    hash_table = []
    for bands in range(bucket_size):
        hash_table.append(defaultdict(list))
        oband = signature_mat[rows*bands:rows*(bands+1)]
        for user in range(num_of_users):
            hashes = tuple(oband[:,user])
            hash_table[bands][hashes].append(user)
    return hash_table



def LSH_minhash(signature_mat, sparse_mat, Perm, Bucket, start, threshold):
    """
    Performs the LSH in order to find the pairs
    """
   
    S = time.time()
    
    hash_table = create_hash_table(signature_mat,permutes=Perm,bucket_size=Bucket)
   
    
    time_lapse("creating the hash table",S,time.time())
     
   
    S = time.time()
    
    pot_cand_pair = set() 
    candidates = set()  
    
    user_cap = 700
   
    for j, var_hash in enumerate(hash_table):
        
        sim_pairs = set()
        
        for i, key in enumerate(list(var_hash.keys())[:int(len(var_hash.keys()))]):
            ## Time cap to exit    
            if round((time.time()-start)/60,ndigits=1)>14.5:
                break
            
            users = np.array(var_hash[key])
                     
            ## Skipping the hashes with only one user
            if(len(users)==1):
                continue
            
            ## Limiting the number of users to pick from a has table    
            if len(users) > user_cap:
                np.random.shuffle(users)
                users = users[:user_cap]
            
            ## For each combination of candidate pair found after minhashing in the hash table
            for i, combi in enumerate(combinations(users, 2)):
                if combi in pot_cand_pair:
                    continue
                pot_cand_pair.add(combi)
                Sim = len(np.where(signature_mat[:,combi[0]]==signature_mat[:,combi[1]])[0])/Perm
                if (Sim >=threshold):
                    sim_pairs.add(combi)
        
        ## For the candidate pairs which have similar signatures we calculate the jaccard similarity
        if len(sim_pairs) > 0:
            CP_1 = np.array(list(sim_pairs))[:,0]
            CP_2 = np.array(list(sim_pairs))[:,1]
            Jaccard_Sim = np.array(jaccardsim(CP_1, CP_2, sparse_mat)).ravel()
            
            ## Candidates with jaccard similarity more than the threshold are saved and written in a file
            if len(np.array(Jaccard_Sim)>=threshold)>0:
                match = np.vstack([[CP_1[Jaccard_Sim>=threshold], CP_2[Jaccard_Sim>=threshold]],Jaccard_Sim[Jaccard_Sim>=threshold]]).T
                for pairs in match:
                    candidates.add(tuple(pairs))
    candidates = np.array(list(candidates), dtype=np.float)
    
    time_lapse("finding the pairs",S,time.time())
    np.savetxt('ans.txt', candidates, fmt='%d,%d,%f',delimiter=' ')                           
    return np.array(list(candidates))




def FindingPairs(file,start,Perm,Bucket,seed,threshold):
    """
    This function finds the pairs and writes it in a file which can be found in the local directory
    """
    mem1 = virtual_memory()
    sparse_mat = create_sparse(filename=file)
    mem2 = virtual_memory()
    signature = create_signature_mat(sparse_mat,permutes=Perm,seed=seed)
    mem3 = virtual_memory()
    matches = LSH_minhash(signature, sparse_mat, Perm, Bucket, start,threshold=threshold)
    mem4 = virtual_memory()
    memp =[mem1.percent,mem2.percent,mem3.percent,mem4.percent]
    memu = [mem1.used/(1024 * 1024), mem2.used/(1024 * 1024),mem3.used/(1024 * 1024),mem4.used/(1024 * 1024)]
    memf = [mem1.free/(1024 * 1024), mem2.free/(1024 * 1024),mem3.free/(1024 * 1024),mem4.free/(1024 * 1024)]
    print("Number of pairs found with similarity greater than %f is %d" % (threshold, len(matches)))
    return matches,memp,memu,memf



file = "user_movie.npy"
Pairs,memp,memu,memf = FindingPairs(file,start,Perm = 120,Bucket = 24,seed = 23,threshold = 0.5)
time_lapse("on entire thing",start,time.time())
memu.append(0)
memp.append(max(memp))
memf.append(max(memf))
memu.sort()
memf.sort(reverse=True)
memp.sort()



plt.plot(memp,memf,'ro' , label='Functions')
plt.plot(memp,memf,'red' , label='Memory available')
plt.plot(memp,memu ,'go' , label = 'Functions')
plt.plot(memp,memu ,'green' , label = 'Memory used')
plt.ylabel('memmory in KB')
plt.xlabel('percentage of memory')
plt.title('Memory available vs used with each function')
plt.legend()
plt.show()